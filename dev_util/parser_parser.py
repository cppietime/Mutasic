'''Generate LR parse tables'''

from dataclasses import dataclass

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
    
    def __getitem__(self, key):
        for terminal in self.terminals:
            if terminal.name == key:
                return terminal
        for nonterminal in self.nonterminals:
            if nonterminal.name == key:
                return nonterminal
    
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
    
    def closure(self, state, firsts, follows):
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
    
    def progress_state(self, state, transition, firsts, follows):
        items = {}
        for old_item, old_la in state.items.items():
            if old_item.next_symbol() != transition:
                continue
            items.setdefault(old_item.next_item(), set()).update(old_la)
        return self.closure(LR1State(items), firsts, follows)
    
    def collect_next_states(self, state, firsts, follows):
        next_symbols = set()
        for item in state.items.keys():
            nsym = item.next_symbol()
            if nsym:
                next_symbols.add(nsym)
        next_states = []
        for nsym in next_symbols:
            next_states.append(self.progress_state(state, nsym, firsts, follows))
        return tuple(next_states)
    
    def collect_states(self, firsts, follows):
        states = {}
        initial_state = self.initial_state()
        initial_state = self.closure(initial_state, firsts, follows)
        states[initial_state.core()] = initial_state
        changed = True
        while changed:
            changed = False
            for core in tuple(states.keys()):
                next_states = self.collect_next_states(states[core], firsts, follows)
                for state in next_states:
                    core = state.core()
                    if core in states:
                        changed |= states[core].changes_on_merge_lalr(state)
                        states[core] = states[core].merge_lalr(state)
                    else:
                        changed = True
                        states[core] = state
        return states

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