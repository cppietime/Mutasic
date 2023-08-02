'''Test the produced parsers'''

from abc import ABC
import io
import json
import sys

from . import ebnf_parser, token

class TableParser:
    def __init__(self, jsonobj):
        self.terminal_names = {}
        self.terminal_ids = {}
        self.nonterminal_names = {}
        self.nonterminal_ids = {}
        self.rules = {}
        for sym in jsonobj['terminals']:
            name = sym['name']
            num = sym['id']
            self.terminal_ids[num] = name
            self.terminal_names[name] = num
        for sym in jsonobj['nonterminals']:
            name = sym['name']
            num = sym['id']
            self.nonterminal_ids[num] = name
            self.nonterminal_names[name] = num
        goal = jsonobj['goal']
        self.goal_id = goal['id']
        self.goal_name = goal['name']
        for rule in jsonobj['rules']:
            rhs = []
            for sym in rule['rhs']:
                rhs.append(sym['name'])
            self.rules[rule['index']] = (rule['lhs']['name'], rhs)
        self.num_states = jsonobj['num_states']
        self.transitions = {i: {} for i in range(self.num_states)}
        for transition in jsonobj['transitions']:
            from_ = transition['from']
            for outgoing in transition['transitions']:
                on = outgoing['on']
                to = outgoing['to']
                self.transitions[from_][on['name']] = to
    
    def parse(self, tokens):
        stack = []
        state = 0
        for token in tokens:
            while token is not None:
                transition = self.transitions[state]
                assert token.token_name() in transition, f'{token.token_name()} not in transitions for {state}\n{stack=}'
                action = transition[token.token_name()]
                if action['action'] == 'SHIFT':
                    stack.append((state, token))
                    state = action['state']
                    token = None
                elif action['action'] == 'REDUCE':
                    rule_num = action['rule_num']
                    rule = self.rules[rule_num]
                    num_syms = len(rule[1])
                    syms = []
                    for _ in range(num_syms):
                        state, sym  = stack.pop()
                        syms.insert(0, sym)
                    goto_rule = self.transitions[state][rule[0]]
                    assert goto_rule['action'] == 'GOTO', f'{goto_rule} is not a GOTO: {state=}, {token.token_name()=}, {stack=}, {rule=}'
                    stack.append((state, (rule[0], syms)))
                    state = goto_rule['state']
                elif action['action'] == 'ACCEPT':
                    token = None
                    return stack
                else:
                    raise Exception(f'Bad state {state} for {token.token_name()}: {action}')
        raise EOFError('Premature EOF!')

def test(rules, text):
    src = io.StringIO()
    src.write(rules)
    src.seek(0)
    grammar = ebnf_parser.load_ebnf(src)
    table = grammar.lalr1_table('goal')
    d = table.to_dict()
    parser = TableParser(d)
    tokens = [NamedToken(x) for x in text.split()]
    return parser.parse(tokens)
    #return d

def test_exprs():
    rules = '''
goal = { stmt } ;
stmt = expr, ";" ;
expr = term, { "+", term } ;
term = factor, { "*", factor } ;
factor = id | "(", expr, ")" | call ;
call = factor, "(", [ expr, { ",", expr } ], ")" ;
'''.strip()
    txt = '''
id "(" id ")" "*" "(" id "+" id ")" ";" id "(" id "," id "+" id "," id "(" ")" ")" ";" "(" id "+" "(" id "*" id ")" ")" "(" ")" ";" $
'''.strip()
    return test(rules, txt)

def load_and_test(rules, text):
    with open(rules, 'r') as file:
        d = json.load(file)
    parser = TableParser(d)
    with open(text, 'r') as file:
        tokens = [token.NamedToken(x) for x in file.read().split()]
    print(parser.parse(tokens))

if __name__ == '__main__':
    load_and_test(sys.argv[1], sys.argv[2])
