'''Generate LR parse tables'''

from dataclasses import dataclass, field
import json

@dataclass(frozen=True)
class TerminalSymbol:
    name: str
    ident: int
    eof: bool = False
    terminal: bool = True
    
    def __repr__(self):
        return f'{self.name}'

@dataclass(frozen=True)
class NonterminalSymbol:
    name: str
    ident: int
    start_symbol: bool = False
    terminal: bool = False
    
    def __repr__(self):
        return f'{self.name}'

@dataclass(frozen=True)
class ProductionRule:
    lhs: NonterminalSymbol
    rhs: tuple
    
    def __len__(self):
        return len(rhs)
    
    def at(self, index=0):
        return LR0Item(self, index)
    
    def __repr__(self):
        return f'{self.lhs.name}->{self.rhs}'

@dataclass(frozen=True)
class LR0Item:
    production_rule: ProductionRule
    position: int
    
    def next_item(self):
        return LR0Item(self.production_rule, self.position + 1)
    
    def finished(self):
        return self.position == len(self.production_rule.rhs)
    
    def next_symbol(self):
        if self.finished():
            return None
        return self.production_rule.rhs[self.position]
    
    def __repr__(self):
        return f'{self.production_rule}@{self.position}'

@dataclass(frozen=True)
class LR1Item:
    lr0_item: LR0Item
    lookahead: frozenset[TerminalSymbol]
    
    def next_item(self):
        return LR1Item(self.lr0_item.next_item(), self.lookahead)
    
    def __repr__(self):
        return f'{self.lr0_item}:{self.lookahead}'

@dataclass
class LR1State:
    items: dict[LR0Item, set[TerminalSymbol]]
    
    def core(self):
        return frozenset(self.items.keys())
    
    def merge_lalr(self, other):
        assert self.core() == other.core()
        items = {}
        for item, sl in self.items.items():
            ol = other.items[item]
            items[item] = sl.union(ol)
        return LR1State(items)
    
    def changes_on_merge_lalr(self, other):
        for item, sl in self.items.items():
            ol = other.items[item]
            if ol.difference(sl):
                return True
        return False

@dataclass
class Grammar:
    terminals: tuple[TerminalSymbol, ...]
    nonterminals: tuple[NonterminalSymbol, ...]
    rules: tuple[ProductionRule, ...]
    augmented: bool = False
    precedences: dict[TerminalSymbol, int] = field(default_factory = dict)
    
    def __getitem__(self, key):
        for terminal in self.terminals:
            if terminal.name == key:
                return terminal
        for nonterminal in self.nonterminals:
            if nonterminal.name == key:
                return nonterminal
    
    def token_precedence(self, token):
        return self.precedences.get(token, 0)
    
    def rule_precedence(self, rule):
        for token in reversed(rule.rhs):
            if not token.terminal:
                continue
            return self.token_precedence(token)
        return 0
    
    def augment(self, target_symbol):
        if self.augmented:
            return self
        if isinstance(target_symbol, str):
            for sym in self.nonterminals:
                if sym.name == target_symbol:
                    target_symbol = sym
                    break
        eof_symbol = TerminalSymbol('$', len(self.terminals), True)
        start_symbol = NonterminalSymbol('START', len(self.nonterminals), True)
        target_rule = ProductionRule(start_symbol, (target_symbol,))
        return Grammar(self.terminals + (eof_symbol,), self.nonterminals + (start_symbol,), self.rules + (target_rule,), True)
    
    def load_precedences(self, src):
        if isinstance(src, str):
            with open(src, 'r') as file:
                self.load_precedences(file)
            return
        for line in src:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()
            self.precedences[self[a]] = int(b)
    
    def firsts(self):
        sets = {symbol: set() for symbol in self.nonterminals}
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                first = sets[rule.lhs]
                if not rule.rhs:
                    changed |= None not in first
                    first.add(None)
                    continue
                for sym in rule.rhs:
                    if sym.terminal:
                        changed |= sym not in first
                        first.add(sym)
                        break
                    changed |= bool(sets[sym].difference(first))
                    first.update(sets[sym])
                    if None not in sets[sym]:
                        break
        return sets
    
    def follows(self, firsts):
        follows = {symbol: set() for symbol in self.nonterminals}
        follows[self.nonterminals[-1]].add(self.terminals[-1])
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                # A -> xYz -> FIRST(z) in FOLLOW(Y)
                last_sym = None
                epsilon = True
                firsts_so_far = set()
                for sym in reversed(rule.rhs):
                    if not sym.terminal:
                        follow = follows[sym]
                        if last_sym:
                            if last_sym.terminal:
                                changed |= last_sym not in follow
                                follow.add(last_sym)
                            else:
                                changed |= bool(firsts[last_sym].difference({None}).difference(follow))
                                follow.update(firsts[last_sym].difference({None}))
                        else:
                            changed |= bool(follows[rule.lhs].difference(follow))
                            follow.update(follows[rule.lhs])
                        if epsilon:
                            changed |= bool(follows[rule.lhs].difference(follow))
                            follow.update(follows[rule.lhs])
                            changed |= bool(firsts_so_far.difference({None}).difference(follow))
                            follow.update(firsts_so_far.difference({None}))
                            firsts_so_far.update(firsts[sym])
                        if None not in firsts[sym]:
                            epsilon = False
                            firsts_so_far.clear()
                    else:
                        epsilon = False
                    last_sym = sym
        return follows
    
    def initial_state(self):
        assert self.augmented, 'Can only call initial_state on an augmented grammar'
        lr0item = self.rules[-1].at()
        base_state = LR1State({lr0item: {self.terminals[-1]}})
        return base_state
    
    def closure_lalr1(self, state, firsts, follows):
        items = {key: set(val) for key, val in state.items.items()}
        changed = True
        while changed:
            changed = False
            for old_item, old_la in tuple(items.items()):
                nsym = old_item.next_symbol()
                if nsym is None or nsym.terminal:
                    continue
                for rule in self.rules:
                    if rule.lhs != nsym:
                        continue
                    new_la = set()
                    epsilon = True
                    for msym in old_item.production_rule.rhs[old_item.position + 1:]:
                        if msym.terminal:
                            new_la.add(msym)
                            epsilon = False
                        else:
                            new_la.update(firsts[msym].difference({None}))
                            epsilon = None in firsts[msym]
                        if not epsilon:
                            break
                    if epsilon:
                        new_la.update(old_la)
                    new_item = rule.at()
                    old_set = items.setdefault(new_item, set())
                    changed |= bool(new_la.difference(old_set))
                    old_set.update(new_la)
        return LR1State(items)
    
    def progress_state_lalr1(self, state, transition, firsts, follows):
        items = {}
        for old_item, old_la in state.items.items():
            if old_item.next_symbol() != transition:
                continue
            items.setdefault(old_item.next_item(), set()).update(old_la)
        return self.closure_lalr1(LR1State(items), firsts, follows)
    
    def collect_next_states_lalr1(self, state, firsts, follows):
        next_symbols = set()
        for item in state.items.keys():
            nsym = item.next_symbol()
            if nsym:
                next_symbols.add(nsym)
        next_states = {}
        for nsym in next_symbols:
            next_states[nsym] = self.progress_state_lalr1(state, nsym, firsts, follows)
        return next_states
    
    def collect_states_lalr1(self, firsts, follows):
        states_list = []
        cores = {}
        transitions = {}
        initial_state = self.initial_state()
        initial_state = self.closure_lalr1(initial_state, firsts, follows)
        cores[initial_state.core()] = 0
        states_list.append(initial_state)
        changed = True
        while changed:
            changed = False
            for core, state_id in tuple(cores.items()):
                old_state = states_list[state_id]
                next_states = self.collect_next_states_lalr1(old_state, firsts, follows)
                for nsym, state in next_states.items():
                    core = state.core()
                    if core in cores:
                        old_id = cores[core]
                        old_target = transitions.get(state_id, {}).get(nsym, None)
                        if old_target not in (old_id, None):
                            raise Exception('Conflict in creating transition table for state {state_id} on {nsym} between {old_id} and {old_target}!')
                        transitions.setdefault(state_id, {})[nsym] = old_id
                        changed |= states_list[old_id].changes_on_merge_lalr(state)
                        states_list[old_id] = states_list[old_id].merge_lalr(state)
                    else:
                        changed = True
                        cores[core] = len(states_list)
                        transitions.setdefault(state_id, {})[nsym] = len(states_list)
                        states_list.append(state)
        return states_list, cores, transitions
    
    def construct_table(self, states_list, cores, transitions):
        table = {i: {} for i in range(len(states_list))}
        for core, core_id in cores.items():
            state = states_list[core_id]
            outgoing = transitions.get(core_id, {})
            substate = table[core_id]
            for item, lookahead in state.items.items():
                if item.finished():
                    for symbol in lookahead:
                        sym_prec = self.token_precedence(symbol)
                        rule_prec = self.rule_precedence(item.production_rule)
                        action = ('REDUCE', item.production_rule, sym_prec > rule_prec)
                        if symbol in substate:
                            #raise Exception(f'Conflict on state {core_id} for {symbol} between {substate[symbol]} and {action}\nstate={states_list[core_id]}')
                            if sym_prec < rule_prec:
                                substate[symbol] = action
                        else:
                            substate[symbol] = action
            for symbol, goto in outgoing.items():
                if symbol.terminal:
                    action = ('SHIFT', goto)
                    if symbol in substate:
                        prec = substate[symbol][2]
                        if prec:
                            substate[symbol] = action
                        #raise Exception(f'Conflict on state {core_id} for {symbol} between {substate[symbol]} and {action}\nstate={states_list[core_id]}')
                    else:
                        substate[symbol] = action
                else:
                    action = ('GOTO', goto)
                    if symbol in substate:
                        raise Exception(f'Conflict on state {core_id} for {symbol} between {substate[symbol]} and {action}\nstate={states_list[core_id]}')
                    substate[symbol] = action
        return table
    
    def lalr1_table(self, target_symbol=None):
        this = self
        if not this.augmented:
            this = this.augment(target_symbol)
        firsts = this.firsts()
        follows = this.follows(firsts)
        states_list, cores, transitions = this.collect_states_lalr1(firsts, follows)
        table = this.construct_table(states_list, cores, transitions)
        return ShiftReduceTable(this.terminals, this.nonterminals, this.rules, this.nonterminals[-1], table)

@dataclass
class ShiftReduceTable:
    terminals: dict[str, TerminalSymbol]
    nonterminals: dict[str, NonterminalSymbol]
    rules: list[ProductionRule]
    target: NonterminalSymbol
    table: dict[int, dict]
    
    def __post_init__(self):
        rules = dict(map(lambda x: x[::-1], enumerate(self.rules)))
        for from_, tos in self.table.items():
            for on, to in tos.items():
                if to[0] == 'REDUCE':
                    rule = to[1]
                    if rule.lhs == self.target:
                        tos[on] = {'action': 'ACCEPT'}
                    else:
                        key = rules[rule]
                        tos[on] = {'action': 'REDUCE', 'rule_num': key, 'produces': rule.lhs.name, 'consumes': len(rule.rhs)}
                elif to[0] == 'SHIFT':
                    tos[on] = {'action': 'SHIFT', 'state': to[1]}
                elif to[0] == 'GOTO':
                    tos[on] = {'action': 'GOTO', 'state': to[1]}
        self.rules = rules
    
    def to_dict(self):
        return {
            'terminals': [
                {'name': term.name, 'id': term.ident, 'terminal': True}
                for term in self.terminals
            ],
            'nonterminals': [
                {'name': nt.name, 'id': nt.ident, 'terminal': False}
                for nt in self.nonterminals
            ],
            'goal': {'name': self.target.name, 'id': self.target.ident, 'terminal': False},
            'rules': [
                {
                'index': i,
                'lhs':
                    {'name': rule.lhs.name, 'id': rule.lhs.ident, 'terminal': False},
                'rhs': [
                    {'name': sym.name, 'id': sym.ident, 'terminal': sym.terminal}
                    for sym in rule.rhs
                ]
                }
                for i, rule in enumerate(self.rules)
            ],
            'num_states': len(self.table),
            'transitions': [
                {
                    'from': from_,
                    'transitions': [
                        {
                            'on': {'name': on.name, 'id': on.ident, 'terminal': on.terminal},
                            'to': to
                        }
                        for on, to in tos.items()
                    ]
                }
                for from_, tos in self.table.items()
            ]
        }
    
    def __str__(self, **kwargs):
        return json.dumps(self.to_dict(), **kwargs)
    
    def save(self, dst, **kwargs):
        if isinstance(dst, str):
            with open(dst, 'w') as file:
                json.dump(self.to_dict(), file, **kwargs)
            return
        json.dump(self.to_dict(), dst, **kwargs)

def test():
    terminals = (
        TerminalSymbol('id', 0),
        TerminalSymbol('(', 1),
        TerminalSymbol(')', 2),
        TerminalSymbol('+', 3),
    )
    nonterminals = (
        NonterminalSymbol('LIST', 0),
        NonterminalSymbol('EXPR', 1),
        NonterminalSymbol('TERM', 2),
    )
    rules = (
        ProductionRule(nonterminals[0], ()),
        ProductionRule(nonterminals[0], (nonterminals[1], nonterminals[0])),
        ProductionRule(nonterminals[1], (nonterminals[2],)),
        ProductionRule(nonterminals[1], (nonterminals[1], terminals[3], nonterminals[1])),
        ProductionRule(nonterminals[2], (terminals[0],)),
        ProductionRule(nonterminals[2], (terminals[1], nonterminals[1], terminals[2])),
    )
    grammar = Grammar(terminals, nonterminals, rules).augment(nonterminals[0])
    firsts = (grammar.firsts())
    print(firsts)
    print(grammar.follows(firsts))

if __name__ == '__main__':
    test()