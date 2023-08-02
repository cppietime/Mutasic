'''EBNF parser'''

from dev_util import parser_parser, parser_tester

def load_ebnf(src):
    if isinstance(src, str):
        with open(src, 'r') as file:
            tokens = _load_tokens(file)
    else:
        tokens = _load_tokens(src)
    rules_raw, terminals, nonterminals = _organize_rules(tokens)
    ebnf_rules = []
    for rule in rules_raw:
        ebnf_rules.append(_parse_rule(rule, nonterminals))
    replacements = {}
    new_rules = []
    bnf_rules = []
    for i, rule in enumerate(ebnf_rules):
        lhs, rhs = rule
        rhs = _replace_rhs(rhs, replacements, new_rules)
        ebnf_rules[i] = (lhs, rhs)
    nonterminals.update(map(lambda x: x[0], new_rules))
    for rule in ebnf_rules + new_rules:
        lhs, rhs = rule
        if rhs:
            if (rhs[0] == 'concatenation'):
                rhs = tuple(_concat(rhs[1]))
            else:
                rhs = (rhs[1],)
        bnf_rules.append((lhs, rhs))
    terminals = tuple(map(lambda x: parser_parser.TerminalSymbol(*x[::-1]), enumerate(terminals)))
    nonterminals = tuple(map(lambda x: parser_parser.NonterminalSymbol(*x[::-1]), enumerate(nonterminals)))
    terminals_d = dict(map(lambda x: (x.name, x), terminals))
    nonterminals_d = dict(map(lambda x: (x.name, x), nonterminals))
    rules = tuple(map(lambda x: parser_parser.ProductionRule(nonterminals_d[x[0]],
        tuple(map(lambda y: terminals_d.get(y, nonterminals_d.get(y, None)), x[1]))), bnf_rules))
    return parser_parser.Grammar(terminals, nonterminals, rules)

def _concat(rhs):
    terms = []
    for term in rhs:
        if term[0] == 'term':
            terms.append(term[1])
        elif term[0] == 'concatenation':
            terms.extend(_concat(term[1]))
        else:
            raise ValueError('Only term and concatenation should make it here')
    return terms

def _replace_rhs(rhs, replacements, new_rules):
    key = rhs[0]
    if key == 'factor':
        if len(rhs[1]) == 1:
            return rhs
        if len(rhs[1]) == 2:
            modifier = rhs[1][1]
            inner = (_replace_rhs(rhs[1][0], replacements, new_rules), rhs[1][1])
            if inner not in replacements:
                strkey = f'$gnt{len(replacements)}'
                replacements[inner] = strkey
                if modifier == '?':
                    new_rules.append((strkey, ()))
                    new_rules.append((strkey, inner[0]))
                elif modifier == '*':
                    new_rules.append((strkey, ()))
                    new_rules.append((strkey, ('concatenation', (('term', strkey), inner[0]))))
                elif modifier == '+':
                    new_rules.append((strkey, inner[0]))
                    new_rules.append((strkey, ('concatenation', (('term', strkey), inner[0]))))
            return ('term', replacements[inner])
        else:
            raise NotImplementedError('Minus terms not yet implemented')
    elif key == 'alternation':
        if len(rhs[1]) == 1:
            return _replace_rhs(rhs[1][0], replacements, new_rules)
        inner = tuple(map(lambda x: _replace_rhs(x, replacements, new_rules), rhs[1]))
        if inner not in replacements:
            strkey = f'$gnt{len(replacements)}'
            replacements[inner] = strkey
            for repl in inner:
                new_rules.append((strkey, repl))
        return ('term', replacements[inner])
    elif key == 'concatenation':
        concs = []
        for conc in rhs[1]:
            concs.append(_replace_rhs(conc, replacements, new_rules))
        return ('concatenation', tuple(concs))
    return rhs

def _parse_rule(raw_rule, nonterminals):
    assert raw_rule[1] == '=' and raw_rule[-1] == ';', 'Malformed rule'
    rhs = _parse_rhs(raw_rule[2:-1])[:2]
    return (raw_rule[0], rhs)

def _parse_rhs(toks):
    concs = []
    i = 0
    while i < len(toks):
        conc = _parse_conc(toks[i:])
        if not conc:
            break
        i += conc[2]
        concs.append(conc[:2])
        if i < len(toks) and toks[i] != '|':
            break
        i += 1
    if len(concs) == 1:
        return concs[0][:2] + (i,)
    return ('alternation', tuple(concs), i)

def _parse_conc(toks):
    facts = []
    i = 0
    while i < len(toks):
        fact = _parse_fact(toks[i:])
        if not fact:
            break
        i += fact[2]
        facts.append(fact[:2])
        if i < len(toks) and toks[i] != ',':
            break
        i += 1
    if len(facts) == 1:
        return facts[0][:2] + (i,)
    return ('concatenation', tuple(facts),i)

def _parse_fact(toks):
    terms = []
    term = _parse_term(toks)
    if not term:
        return None
    terms.append(term[:2])
    i = term[2]
    if i < len(toks):
        if toks[i] in {'+', '*', '?'}:
            terms.append(toks[i])
            i += 1
        elif toks[i] == '-':
            terms.append(toks[i])
            t2 = _parse_term(toks[i+1:])
            i += 1 + t2[2]
            assert t2, 'Malformed factor, term required after -'
            terms.append(t2[:2])
    if len(terms) == 1:
        return terms[0][:2] + (i,)
    elif len(terms) == 2 and terms[0][0] == 'factor':
        child = terms[0][1]
        if len(child) == 1:
            return ('factor', (child[0], term[1]), i)
        elif len(child) == 2:
            a = terms[1]
            b = child[1]
            if a == b:
                return ('factor', (child[0], a), i)
            return ('factor', (child[0], '*'), i)
    return ('factor', tuple(terms), i)

def _parse_term(toks):
    i = 0
    if toks[0] in {'(', '{', '['}:
        inner = _parse_rhs(toks[1:])
        i = inner[2] + 2
        assert len(toks) > i - 1, f'No matching end token in {toks}, consumed {inner}'
        end = toks[i - 1]
        ending = {'(': ')', '{': '}', '[': ']'}
        assert end == ending[toks[0]], f'Malformed term, begin and end do not match. {toks=}, {end=}'
        if toks[0] == '(':
            return inner[:2] + (i,)
        elif toks[0] == '{':
            return ('factor', (inner[:2], '*'), i)
        return ('factor', (inner[:2], '?'), i)
    return ('term', toks[0], 1)

def _organize_rules(tokens):
    rules = []
    rule = []
    terminals = set()
    nonterminals = set()
    for token in tokens:
        if token.startswith('(*'):
            continue
        if not rule:
            nonterminals.add(token)
        else:
            if token[0] not in {',', '|', '=', ';', '[', '{', '?', ']', '}', '(', ')', '+', '*', '-'}:
                terminals.add(token)
            if len(rule) == 1:
                assert token == '=', 'Expected assignment'
        rule.append(token)
        if token == ';':
            assert len(rule) >= 3, 'Premature semicolon'
            rules.append(rule)
            rule = []
    terminals.difference_update(nonterminals)
    return rules, terminals, nonterminals

def _load_tokens(src):
    tokens = []
    buffer = []
    state = 'start'
    while True:
        c = src.read(1)
        if not c:
            break
        if state == 'start':
            if c in {',', '|', '=', ';', '[', '{', '?', ']', '}', '?', '+', '*', '-'}:
                if buffer:
                    tokens.append(''.join(buffer).strip())
                tokens.append(c)
                buffer.clear()
            elif c == '(':
                if buffer:
                    tokens.append(''.join(buffer).strip())
                buffer.clear()
                buffer.append(c)
                state = 'lparen'
            elif c == '"':
                if buffer:
                    tokens.append(''.join(buffer).strip())
                buffer.clear()
                buffer.append(c)
                state = 'dquote'
            elif c == "'":
                if buffer:
                    tokens.append(''.join(buffer).strip())
                buffer.clear()
                buffer.append(c)
                state = 'squote'
            elif c.isspace():
                if buffer:
                    tokens.append(''.join(buffer).strip())
                buffer.clear()
            else:
                buffer.append(c)
        elif state == 'lparen':
            if c == '*':
                buffer.append('*')
                state = 'comment'
            else:
                tokens.append('(')
                buffer.clear()
                state = 'start'
        elif state == 'comment':
            buffer.append(c)
            if c == '*':
                state = 'comstar'
        elif state == 'comstar':
            buffer.append(c)
            if c == ')':
                tokens.append(''.join(buffer).strip())
                buffer.clear()
                state = 'start'
            else:
                state = 'comment'
        elif state == 'dquote':
            if c == '\\':
                state == 'dqbs'
            elif c == '"':
                buffer.append(c)
                tokens.append(''.join(buffer).strip())
                buffer.clear()
                state = 'start'
            else:
                buffer.append(c)
        elif state == 'dqbs':
            special = {'n': '\n', 't': '\t', '"': '"', '\\': '\\', 'r': '\r'}
            if c in special:
                buffer.append(special[c])
            else:
                buffer += ['\\', c]
            state = 'dquote'
        elif state == 'squote':
            if c == '\\':
                state == 'sqbs'
            elif c == "'":
                buffer.append(c)
                tokens.append(''.join(buffer).strip())
                buffer.clear()
                state = 'start'
            else:
                buffer.append(c)
        elif state == 'sqbs':
            special = {'n': '\n', 't': '\t', "'": "'", '\\': '\\', 'r': '\r'}
            if c in special:
                buffer.append(special[c])
            else:
                buffer += ['\\', c]
            state = 'squote'
    return tokens

def test():
    import io
    src = io.StringIO()
    src.write('''
program = { expr } ;
expr = (* An expression *) term, { "+", term } ;
term = factor, { "*", factor } ;
factor = id | "(", expr, ")" ;
    '''.strip())
    src.seek(0)
    grammar = load_ebnf(src)
    table = grammar.lalr1_table('program')
    d = table.to_dict()
    print(table.__str__(indent=1))
    parser = parser_tester.TableParser(d)
    
    tokens = [parser_tester.NamedToken(x) for x in 'id "*" "(" id "+" id ")" id id $'.split()]
    parse = parser.parse(tokens)
    print(parse)

if __name__ == '__main__':
    test()