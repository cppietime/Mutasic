from abc import ABC

import spyceballs as sb

from . import compiler

# Orchestra AST data types

class HasValue(ABC):
    """Abstract class for AST nodes that can be evaluated."""
    def is_constant(self, ctx):
        return NotImplemented
    
    def type(self, ctx):
        return NotImplemented
    
    def rate(self, ctx):
        return NotImplemented

class ForwardScannable(ABC):
    def scan_scope(self, ctx, scope):
        return NotImplemented

class Tacable(ABC):
    def eval(self, ctx, scope):
        return NotImplemented

class Assignable(ABC):
    def assign_to(self, ctx, scope, value):
        return NotImplemented

class Funcdef(ForwardScannable):
    def __init__(self, r):
        self.return_type = r[0]
        self.name = r[1]
        self.body = r[3]
        if not r[2]:
            self.params = []
        else:
            self.params = r[2][0]
        print(self.params)
    
    def scan_scope(self, ctx, _):
        partype_names = tuple(map(lambda p: p[0].name, self.params))
        sig = self.return_type.name + '(' + ','.join(partype_names) + ')'
        if sig not in ctx.types:
            functype = ctx.types[sig] = compiler.Type(sig, {}, {}, True, return_type=self.return_type, args_types=tuple(map(lambda p: ctx.types[p], partype_names)))
        else:
            functype = ctx.types[sig]
        ctx.funcs[(self.name, partype_names)] = compiler.Function(self.name, self.name, functype, self)

class Vardec(ForwardScannable, Tacable):
    def __init__(self, r):
        self.type_ = r[0]
        self.vars = []
        for v in r[1]:
            name = v[0]
            value = v[1][1] if len(v) == 2 else None
            self.vars.append((name, value))
    
    def scan_scope(self, ctx, scope):
        for name, init in self.vars:
            if name in scope:
                raise SyntaxError(f'Redefinition of variable {name} shadows previously declared variable')
            print(f'{name=} {init=} {init.is_constant(ctx)=}')
            scope[name] = compiler.Variable(name, self.type_.type(ctx), False, init)
            ctx.vars[name] = scope[name]
    
    def eval(self, ctx, scope):
        self.scan_scope(ctx, scope)
        tacs = []
        # Evaluate assigned expressions
        for name, init in self.vars:
            if init is None:
                continue
            tacs += init.eval(ctx, scope)
            tacs.append(f'pop {self.type_.name} {name};')
        return tacs

class LeftAssocBinOp(HasValue, Tacable):
    def __init__(self, r):
        self.children = [r[0]] + [x[1] for x in r[1]]
        self.ops = [x[0] for x in r[1]]
        if len(self.children) != len(self.ops) + 1:
            raise ValueError(f'Mismatch in number of child expressions and binary operators: {r}')
    
    def is_constant(self, ctx):
        return all(map(lambda e: e.is_constant(ctx), self.children))
    
    def type(self, ctx):
        t = self.children[0].type(ctx)
        if t is None:
            print(f'{self.children[0]=}')
        return t
    
    def eval(self, ctx, _):
        tacs = self.children[0].eval(ctx, _)
        lasttype = self.children[0].type(ctx)
        for child, op in zip(self.children[1:], self.ops):
            childtype = child.type(ctx)
            common_type = ctx.common_type(lasttype, childtype)
            if common_type is None:
                raise TypeError(f'No way to coerce types {lasttype} with {childtype}')
            if lasttype != common_type:
                tacs.append(f'cast {lasttype.name} -> {common_type.name};')
            tacs += child.eval(ctx, _)
            if childtype != common_type:
                tacs.append(f'cast {childtype.name} -> {common_type.name};')
            tacs.append(f'{op} {common_type.name} {common_type.name};')
            lasttype = common_type
        return tacs

class Accessor(HasValue, Tacable, Assignable):
    def __init__(self, r):
        base, resolver = r
        if resolver[-1][0] == '.':
            self.type_ = 'FIELD'
            self.selector = resolver[-1][1]
        elif resolver[-1][0] == '[':
            self.type_ = 'ELEMENT'
            self.selector = resolver[-1][1]
        else:
            raise ValueError(f'Invalid accessor argument: {resolver[-1]}')
        if len(r) == 2:
            self.parent = r[0]
        else:
            child = resolver[-2]
            if child[0] == '(':
                self.parent = Call([r[0], resolver[:-1]])
            else:
                self.parent = Accessor([r[0], resolver[:-1]])
    
    def is_constant(self, ctx):
        if self.type_ == 'FIELD':
            pt = self.parent.type(ctx)
            child = pt.fields.get(self.selector, pt.methods.get(self.selector, None))
            if child is None:
                raise KeyError(f'No such member {self.selector} exists')
            ct = child.type
            return self.parent.is_constant(ctx) and (ct.is_function or ct.is_constant)
        return self.parent.is_constant(ctx) and self.selector.is_constant(ctx)
    
    def type(self, ctx):
        pt = self.parent.type(ctx)
        if self.type_ == 'ELEMENT':
            if pt.base_type is None:
                raise TypeError('Cannot index un-indexable type')
            return pt.base_type
        return pt.fields[self.selector].type
    
    def eval(self, ctx, _):
        tacs = self.parent.eval(ctx, _)
        if self.type_ == 'ELEMENT':
            tacs += self.selector.eval(ctx, _)
            tacs.append('index;')
        else:
            tacs.append(f'access {self.selector};')
        return tacs
    
    def assign_to(self, ctx, _, value):
        tacs = self.parent.eval(ctx, _)
        tacs += self.selector.eval(ctx, _)
        mytype = self.type(ctx)
        valtype = value.type(ctx)
        if mytype != valtype:
            casttype = ctx.cast_to(valtype, mytype)
            if casttype is None:
                raise TypeError(f'Cannot implicitly cast type {valtype.name} to {mytype.name}')
            tacs.append(f'cast {valtype.name} -> {casttype.name};')
        tacs += value.eval(ctx, _)
        if self.type_ == 'ELEMENT':
            tacs.append('indexed_write;')
        else:
            tacs.append(f'field_write {self.selector};')
        return tacs

class Call(HasValue, Tacable):
    def __init__(self, r):
        base, resolver = r
        self.args = resolver[-1][1:-1]
        if len(r) == 2:
            self.function = r[0]
        else:
            child = r[1][-2]
            if child[0] == '(':
                self.function = Call([r[0], resolver[:-1]])
            else:
                self.function = Accessor([r[0], resolver[:-1]])
    
    def is_constant(self, ctx):
        return False
    
    def type(self, ctx):
        pt = self.function.type(ctx)
        if not pt.is_function:
            raise TypeError('Attempt to call non-function type')
        return pt.return_type
    
    def eval(self, ctx, _):
        if not isinstance(self.function, Variable):
            raise ValueError('Function calls currently only allowed on direct named functions')
        name = self.function.name
        types = [arg.type(ctx) for arg in self.args]
        tacs = []
        for arg in self.args:
            tacs += arg.eval(ctx, _)
        tacs.append(f'call {name} {[arg.name for arg in types]};')
        return tacs

class Assignment(HasValue, Tacable):
    def __init__(self, r):
        lhs, self.value = r
        self.targets = [v[0] for v in lhs]
        self.ops = [v[1] for v in lhs]
        if len(self.targets) != len(self.ops):
            raise ValueError(f'Mismatch in number of child expressions and binary operators: {r}')
    
    def is_constant(self, ctx):
        return False
    
    def type(self, ctx):
        lhs, rhs = self.targets[0], (self.targets[1] if len(self.targets) > 2 else self.value)
        lt, rt = lhs.type(ctx), rhs.type(ctx)
        # if self.ops[0] == '=':
            # return lt
        # op = self.ops[0][:-1]
        # return lt.methods[(op, rt)].return_type
        return lt
    
    def eval(self, ctx, _):
        sources = self.targets[1:] + [self.value]
        tacs = []
        for target, source, op in zip(self.targets, sources, self.ops):
            if op == '=':
                tacs += target.assign_to(ctx, _, source)
            else:
                binop = LeftAssocBinOp([target, [op[0], source]])
                tacs += target.assign_to(ctx, _, binop)
        return tacs

class Variable(HasValue, Tacable, Assignable):
    def __init__(self, r):
        self.name = r
    
    def is_constant(self, ctx):
        if self.name in ctx.vars:
            raise KeyError(f'Variable {self.name} not defined')
        return ctx.vars[self.name].is_constant
    
    def type(self, ctx):
        if self.name in ctx.vars:
            raise KeyError(f'Variable {self.name} not defined')
        return ctx.vars[self.name].type
    
    def eval(self, ctx, _):
        return [f'push variable {self.name};']
    
    def assign_to(self, ctx, _, value):
        tacs = value.eval(ctx, _)
        mytype = self.type(ctx)
        valtype = value.type(ctx)
        if mytype != valtype:
            casttype = ctx.cast_to(valtype, mytype)
            if casttype is None:
                raise TypeError(f'Cannot implicitly cast type {valtype.name} to {mytype.name}')
            tacs.append(f'cast {valtype.name} -> {casttype.name};')
        tacs.append(f'pop {self.name};')
        return tacs

class Constant(HasValue, Tacable):
    def __init__(self, r):
        if r.startswith('"'):
            self.type_ = 'string'
            self.value = r
        else:
            self.type_ = 'float'
            self.value = float(r)
    
    def is_constant(self, ctx):
        return True
    
    def type(self, ctx):
        return ctx.num_type
    
    def eval(self, ctx, _):
        return [f'push const {self.value};']

class UnaryOp(HasValue, Tacable):
    def __init__(self, r):
        self.value = r[-1]
        if len(r) == 2:
            self.op = r[0]
        else:
            self.op = None
    
    def is_constant(self, ctx):
        return self.value.is_constant(ctx)
    
    def type(self, ctx):
        return self.value.type(ctx)
    
    def eval(self, ctx, _):
        tacs = self.value.eval(ctx, _)
        if self.op is not None:
            tacs.append(f'{self.op} {self.type(ctx)}')
        return tacs

class Return(Tacable):
    def __init__(self, r):
        if len(r) == 1:
            self.value = None
        self.value = r[1]
    
    def eval(self, ctx, _):
        if self.value is None:
            return ['return nil;']
        tacs = self.value.eval(ctx, _)
        tacs.append('return pop;')
        return tacs

class If(Tacable):
    def __init__(self, r):
        self.predicate = r[1]
        self.if_true = r[2]
        if len(r) == 4:
            self.if_false = r[3][1]
    
    def eval(self, ctx, _):
        tacs = self.predicate.eval(ctx, _)
        true_block = self.if_true.eval(ctx, _)
        false_block = []
        if self.if_false is not None:
            false_block = self.if_false.eval(ctx, _)
            true_block.append(f'jmp back {len(false_block)};')
        tacs.append(f'if false jmp ahead {len(true_block)};')
        return tacs + true_block + false_block

class For(Tacable):
    def __init__(self, r):
        self.before = r[1]
        self.predicate = r[2]
        self.after_each = r[3]
        self.statement = r[4]
    
    def eval(ctx, _):
        tacs = self.before.eval(ctx, _)
        condition = self.predicate.eval(ctx, _)
        body = self.statement.eval(ctx, _)
        after = self.after.eval(ctx, _)
        tacs += condition
        tacs.append(f'if false jmp ahead {len(body) + len(after) + 1};')
        tacs += body
        tacs += after
        tacs.append(f'jmp back {len(after) + len(body) + 2};')
        return tacs

class While(Tacable):
    def __init__(self, r):
        self.predicate = r[1]
        self.statement = r[2]
    
    def eval(self, ctx, _):
        tacs = self.predicate.eval(ctx, _)
        body = self.statement.eval(ctx, _)
        tacs.append(f'if false jmp ahead {len(body) + 1}')
        tacs += body
        tacs.append('jmp back {len(body) + 1};')
        return tacs

# TODO figure out code for break and continue
class Break:
    def __init__(self, r):
        pass

class Continue:
    def __init__(self, r):
        pass

class Typespec:
    def __init__(self, r):
        self.name = r[0]
        self.array = len(r[1]) if len(r) == 2 else 0
    
    def type(self, ctx):
        key = self.name
        if self.array:
            key += '[]'
        return ctx.types[key]

class ExprStmt(Tacable):
    def __init__(self, r):
        self.expr = r[0]
    
    def eval(self, ctx, _):
        tacs = self.expr.eval(ctx, _)
        tacs.append('pop nil;')
        return tacs

class BlockStmt:
    def __init__(self, r):
        self.statements = r[0]
    
    def eval(self, ctx, _):
        return ctx.eval_block(self)
