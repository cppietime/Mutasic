import sys

from dev_util import ebnf_parser

def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    grammar = ebnf_parser.load_ebnf(src)
    if len(sys.argv) > 3:
        grammar.load_precedences(sys.argv[3])
    table = grammar.lalr1_table('goal')
    table.save(dst, indent=2)

if __name__ == '__main__':
    main()
