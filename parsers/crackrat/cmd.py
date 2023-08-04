"""CLI command of module. For now, just generates barely functional
parser tables.
"""

import sys

from . import ebnf_parser

def genlr():
    src = sys.argv[1]
    dst = sys.argv[2]
    grammar = ebnf_parser.load_ebnf(src)
    if len(sys.argv) > 3:
        grammar.load_precedences(sys.argv[3])
    table = grammar.lalr1_table('goal')
    table.save(dst, indent=2)

cmds = {
    'genlr': genlr
}

def main():
    cmd = sys.argv[1]
    if cmd in cmds:
        cmds[cmd]()
    else:
        raise ValueError(f'Unrecognized command: {cmd}.\nTry one of {"\n".join(cmds.keys())}')

if __name__ == '__main__':
    main()
