"""A lexer that tokenizes strings by regexes."""

import re

from . import token

class RegexLexer:
    def __init__(self, rules):
        self.rules = {}
        for production, rule in rules.items():
            self.rules[production] = re.compile(rule)
    
    def split_parse(self, src, filename=''):
        """Eat up a string and return all the tokens comprising it."""
        line_no = 0
        col_no = 0
        i = 0
        tokens = []
        while i < len(src):
            best = None
            for production, rule in self.rules.items():
                m = rule.match(src[i:])
                if not m:
                    continue
                g = m.group(0)
                size = len(g)
                if best is None or size > best[0]:
                    best = (size, production)
            if best is None:
                raise ValueError(f'No token found at position {i}: "{src[i:i+10]}"')
            size, symbol = best
            lexeme = src[i:i+size]
            tokens.append(token.StaticToken(symbol, src[i:i+size], line_no, col_no, filename))
            if '\n' in lexeme:
                line_no += lexeme.count('\n')
                col_no = size - lexeme.rfind('\n')
            else:
                col_no += size
            i += size
        return tokens
    
    @staticmethod
    def load(src):
        """Load rules from a file of the form:
        symbol_name := regex_to_match ; comment
        Blank lines are ignored.
        """
        if isinstance(src, str):
            with open(src, 'r') as file:
                return RegexLexer.load(file)
        rules = {}
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith(';'):
                continue
            left, right = line.split(':=')
            symname = left.strip()
            regex = right.strip()
            if regex.contains(';'): # Comment
                regex, _ = regex.split(';')
                regex = regex.strip()
            rules[symname] = regex
        return RegexLexer(rules)

def test():
    rules = {
        'int': '-?\\d+',
        'id': '\\w(\\w|\\d| _)*',
        'lparen': '\\(',
        'rparen': '\\)',
        'ws': '\\s+'
    }
    lexer = RegexLexer(rules)
    tokens = lexer.split_parse('''
hello world 1234 (5)
    ''')
    for token in tokens:
        print(token)

if __name__ == '__main__':
    test()
