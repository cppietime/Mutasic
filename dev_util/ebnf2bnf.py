'''Convert EBNF to BNF'''

from dev_util import ebnf_parser, parser_tester, parser_parser

def load_ebnf_lrgram(src):
    ebnf_rules, terminals, nonterminals = ebnf_parser.load_ebnf_raw(src)
    bnf_rules, new_nonterminals = ebnf2bnf(ebnf_rules)
    nonterminals.update(new_nonterminals)
    terminals = tuple(map(lambda x: parser_parser.TerminalSymbol(*x[::-1]), enumerate(terminals)))
    nonterminals = tuple(map(lambda x: parser_parser.NonterminalSymbol(*x[::-1]), enumerate(nonterminals)))
    terminals_d = dict(map(lambda x: (x.name, x), terminals))
    nonterminals_d = dict(map(lambda x: (x.name, x), nonterminals))
    rules = tuple(map(lambda x: parser_parser.ProductionRule(nonterminals_d[x[0]],
        tuple(map(lambda y: terminals_d.get(y, nonterminals_d.get(y, None)), x[1]))), bnf_rules))
    return parser_parser.Grammar(terminals, nonterminals, rules)

def ebnf2bnf(ebnf_rules):
    replacements = {}
    new_rules = []
    bnf_rules = []
    new_nonterminals = set()
    for i, rule in enumerate(ebnf_rules):
        lhs, rhs = rule
        rhs = _replace_rhs(rhs, replacements, new_rules)
        ebnf_rules[i] = (lhs, rhs)
    new_nonterminals.update(map(lambda x: x[0], new_rules))
    for rule in ebnf_rules + new_rules:
        lhs, rhs = rule
        if rhs:
            if (rhs[0] == 'concatenation'):
                rhs = tuple(_concat(rhs[1]))
            else:
                rhs = (rhs[1],)
        bnf_rules.append((lhs, rhs))
    return bnf_rules, new_nonterminals

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
    grammar = load_ebnf_lrgram(src)
    print(grammar)
    table = grammar.lalr1_table('program')
    d = table.to_dict()
    print(table.__str__(indent=1))
    parser = parser_tester.TableParser(d)
    
    tokens = [parser_tester.NamedToken(x) for x in 'id "*" "(" id "+" id ")" id id $'.split()]
    parse = parser.parse(tokens)
    print(parse)

if __name__ == '__main__':
    test()
