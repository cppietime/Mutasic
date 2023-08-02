'''EBNF parser'''

def load_ebnf_raw(src):
    '''src: a filename or read-able object.
    Parses the input according to EBNF grammar.
    Returns a 3-tuple of form (rules, terminals, nonterminals).
    rules: A list of rules, each a 2-tuple of (left-hand side, right-hand side).
    terminals: A set of all found symbols for which there is no rule.
    nonterminals: A set of all found sybmols for which there is a rule.
    '''
    if isinstance(src, str):
        with open(src, 'r') as file:
            tokens = _load_tokens(file)
    else:
        tokens = _load_tokens(src)
    rules_raw, terminals, nonterminals = _organize_rules(tokens)
    ebnf_rules = []
    for rule in rules_raw:
        ebnf_rules.append(_parse_rule(rule, nonterminals))
    return ebnf_rules, terminals, nonterminals

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
                assert token == '=', f'Invalid rule: {" ".join(rule)}: Expected assignment'
        rule.append(token)
        if token == ';':
            assert len(rule) >= 3, f'Invalid rule: {" ".join(rule)}:Premature semicolon'
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
            if c in {',', '|', '=', ';', '[', '{', '?', ']', '}', '?', '+', '*', '-', ')'}:
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
    print(load_ebnf_raw(src))

if __name__ == '__main__':
    test()