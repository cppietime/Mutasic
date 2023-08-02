'''Packrat parser'''

from dataclasses import dataclass
import io
from . import ebnf_parser, token

@dataclass
class PackratAST:
    lhs: str
    rhs: tuple
    start: int
    end: int
    tree: tuple = ()
    
    def __post_init__(self):
        flat = []
        for subtree in self.tree:
            if not isinstance(subtree, PackratAST):
                flat.append(subtree)
                continue
            if subtree.lhs is None:
                flat += subtree.tree
            else:
                flat.append(subtree)
        self.tree = tuple(flat)
    
    def __str__(self, indents=0, tab=2):
        s = f'{" " * indents}{self.lhs} [{self.start}, {self.end})\n'
        for subtree in self.tree:
            if isinstance(subtree, PackratAST):
                s += subtree.__str__(indents+tab, tab)
            else:
                s += f'{" " * (indents + tab)}{subtree}\n'
        return s

class PackratParser:
    '''Parses set of EBNF rules with Packrat parsing'''
    def __init__(self, rules):
        self.rules = {}
        for rule in rules:
            self.rules.setdefault(rule[0], []).append(rule) # Use a list, order matters
        self.memo = None
    
    def parse(self, goal, tokens):
        assert goal in self.rules, f'No rule(s) to build {goal}'
        self.memo = {} # Form (rule, pos): result | None
        self.recurse = set() # Form (symbol, pos): [limit, sofar]
        return self._parse_symbol(goal, tokens, 0)
    
    def _parse_symbol(self, symbol, tokens, pos):
        '''Results are either None on failure, or (rhs, start, end, subtrees) on success.'''
        key = (symbol, pos)
        if key in self.recurse:
            # We have already started to recurse
            raise Exception(f'Left-recursion found for symbol {symbol}.\nLeft-recursion is not supported, please reformat your grammar.')
        self.recurse.add(key)
        rules = self.rules[symbol]
        result = None
        for rule in rules:
            result = self._parse_rule(rule[1], tokens, pos, rule[0])
            if result is not None:
                break
        self.recurse.remove(key)
        return result
    
    def _parse_rule(self, rule, tokens, pos, lhs):
        '''Memoizing wrapper for _eval_rule.'''
        key = (lhs, rule, pos)
        if key in self.memo:
            return self.memo[key]
        result = self._eval_rule(rule, tokens, pos, lhs)
        self.memo[key] = result
        return result
    
    def _eval_rule(self, rule, tokens, pos, lhs):
        ruletype = rule[0]
        if ruletype == 'term':
            # Base case
            symbol = rule[1]
            if symbol in self.rules:
                # Nonterminal
                result = self._parse_symbol(symbol, tokens, pos)
                if result is not None:
                    return PackratAST(lhs, rule, pos, result.end, (result,))
                return None
            else:
                # Terminal
                if pos < len(tokens) and tokens[pos].token_name() == rule[1]:
                    return PackratAST(lhs, rule, pos, pos + 1, (tokens[pos],))
                return None
        elif ruletype == 'factor':
            if len(rule[1]) == 3:
                # term - term
                assert rule[1][1] == '-', f'Malformed factor {rule[1]}'
                result = self._parse_rule(rule[1][0], tokens, pos, lhs)
                if result is None:
                    return None
                negative = self._parse_rule(rule[1][2], tokens, pos, None)
                if negative is not None:
                    return None
                return result
            elif len(rule[1]) == 2:
                # term [+*?]
                modifier = rule[1][1]
                if modifier == '?':
                    result = self._parse_rule(rule[1][0], tokens, pos, lhs)
                    if result is None:
                        return PackratAST(lhs, rule, pos, pos)
                    return result
                elif modifier in {'*', '+'}:
                    i = pos
                    subtoks = []
                    while True:
                        result = self._parse_rule(rule[1][0], tokens, i, None)
                        if result is None:
                            break
                        i = result.end
                        subtoks.append(result)
                    if modifier == '+' and i == pos:
                        return None
                    return PackratAST(lhs, rule, pos, i, tuple(subtoks))
                raise ValueError(f'Malformed factor {rule[1]}')
        elif ruletype == 'concatenation':
            subtoks = []
            i = pos
            for p, piece in enumerate(rule[1]):
                result = self._parse_rule(piece, tokens, i, None)
                if result is None:
                    return None
                i = result.end
                subtoks.append(result)
            return PackratAST(lhs, rule, pos, i, tuple(subtoks))
        elif ruletype == 'alternation':
            for subrule in rule[1]:
                result = self._parse_rule(subrule, tokens, pos, lhs)
                if result is not None:
                    return result
            return None
        raise ValueError(f'Malformed rule {rule}')

def test():
    src = io.StringIO()
    src.write('''
goal = ( expr, ";" )*, $ ;
expr = term, { "+", expr } ;
term = factor, { "*", term } ;
factor = "(", expr, ")" | funccall | id ;
funccall = id, "(", [ expr, { ",", expr } ], ")" ;
'''.strip())
    src.seek(0)
    rules, terminals, nonterminals = ebnf_parser.load_ebnf_raw(src)
    print(rules,'\n')
    parser = PackratParser(rules)
    result = parser.parse('goal', tuple(map(lambda x: token.StaticToken(x, 0, x), '''
id "(" ")" "*" id "(" ")" ";"
id "*" id "(" id ")" ";"
id "(" id "+" id "," id "(" id ")" ")" "*" id ";"
'''.strip().split() + ['$'])))
    print(f'Result: {result}')

if __name__ == '__main__':
    test()
