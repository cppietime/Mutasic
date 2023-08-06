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

class AOTable(ABC):
    def const_value(self):
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
            mytype = self.type_.type(ctx)
            valtype = init.type(ctx)
            if mytype != valtype:
                casttype = ctx.cast_to(valtype, mytype)
                if casttype is None:
                    raise TypeError(f'Cannot cast {valtype.name} to {mytype.name}')
                tacs.append(f'cast {valtype.name} {mytype.name};')
            tacs.append(f'pop variable {name} {self.type_.name};')
        return tacs

class LeftAssocBinOp(HasValue, Tacable, AOTable):
    def __init__(self, r):
        self.children = [r[0]] + [x[1] for x in r[1]]
        self.ops = [x[0] for x in r[1]]
        if len(self.children) != len(self.ops) + 1:
            raise ValueError(f'Mismatch in number of child expressions and binary operators: {r}')
    
    def is_constant(self, ctx):
        return all(map(lambda e: e.is_constant(ctx), self.children))
    
    def type(self, ctx):
        lasttype = self.children[0].type(ctx)
        for child, op in zip(self.children[1:], self.ops):
            childtype = child.type(ctx)
            binop_types = ctx.binop_types(op, lasttype, childtype)
            if binop_types is None:
                raise TypeError(f'Cannot perform op {op} on types {lasttype.name} and {childtype.name}')
            res_type = binop_types[0]
            lasttype = res_type
        return lasttype
    
    def eval(self, ctx, _):
        cv = self.const_value()
        if cv is not None:
            return [f'push const {cv} {self.type(ctx).name};']
        tacs = self.children[0].eval(ctx, _)
        lasttype = self.children[0].type(ctx)
        for child, op in zip(self.children[1:], self.ops):
            childtype = child.type(ctx)
            binop_types = ctx.binop_types(op, lasttype, childtype)
            if binop_types is None:
                raise TypeError(f'Cannot perform op {op} on types {lasttype.name} and {childtype.name}')
            res_type, l_type, r_type = binop_types
            if lasttype != l_type:
                tacs.append(f'cast {lasttype.name} {l_type.name};')
            tacs += child.eval(ctx, _)
            if childtype != r_type:
                tacs.append(f'cast {childtype.name} {r_type.name};')
            tacs.append(f'{op} {l_type.name} {r_type.name} {res_type.name};')
            lasttype = res_type
        if lasttype != self.type(ctx):
            # Should never actually occur
            tacs.append(f'cast {lasttype.name} {self.type(ctx).name}')
        return tacs
    
    def const_value(self):
        lvalue = None
        for i, child in enumerate(self.children):
            if not isinstance(child, AOTable):
                return None
            value = child.const_value()
            if value is None:
                return None
            if lvalue == None:
                lvalue = value
            else:
                rvalue = value
                lvalue = {
                    '+':  lambda lvalue, rvalue: lvalue + rvalue,
                    '-':  lambda lvalue, rvalue: lvalue - rvalue,
                    '*':  lambda lvalue, rvalue: lvalue * rvalue,
                    '/':  lambda lvalue, rvalue: lvalue / rvalue,
                    '%':  lambda lvalue, rvalue: lvalue % rvalue,
                    '&':  lambda lvalue, rvalue: lvalue & rvalue,
                    '|':  lambda lvalue, rvalue: lvalue | rvalue,
                    '^':  lambda lvalue, rvalue: lvalue ^ rvalue,
                    '<<': lambda lvalue, rvalue: int(lvalue << rvalue),
                    '>>': lambda lvalue, rvalue: int(lvalue >> rvalue),
                    '==': lambda lvalue, rvalue: int(lvalue == rvalue),
                    '!=': lambda lvalue, rvalue: int(lvalue != rvalue),
                    '>=': lambda lvalue, rvalue: int(lvalue >= rvalue),
                    '<=': lambda lvalue, rvalue: int(lvalue <= rvalue),
                    '>':  lambda lvalue, rvalue: int(lvalue > rvalue),
                    '<':  lambda lvalue, rvalue: int(lvalue < rvalue),
                    '&&': lambda lvalue, rvalue: int(bool(lvalue and rvalue)),
                    '||': lambda lvalue, rvalue: int(bool(lvalue or rvalue))
                }[self.ops[i - 1]](lvalue, rvalue)
        return lvalue

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
            if self.selector.type(ctx) != ctx.types['i1']:
                tacs.append(f'cast {self.selector.type(ctx).name} i1;')
            tacs.append(f'push index {self.type(ctx).name};')
        else:
            ptype = self.parent.type(ctx)
            index = ptype.member_indices[self.selector]
            tacs.append(f'push field {ptype.name} {index} {self.type(ctx).name};')
        return tacs
    
    def assign_to(self, ctx, _, value):
        tacs = self.parent.eval(ctx, _)
        tacs += value.eval(ctx, _)
        mytype = self.type(ctx)
        valtype = value.type(ctx)
        if mytype != valtype:
            casttype = ctx.cast_to(valtype, mytype)
            if casttype is None:
                raise TypeError(f'Cannot implicitly cast type {valtype.name} to {mytype.name}')
            tacs.append(f'cast {valtype.name} {casttype.name};')
        if self.type_ == 'ELEMENT':
            tacs += self.selector.eval(ctx, _)
            if self.selector.type(ctx) != ctx.types['i1']:
                tacs.append(f'cast {self.selector.type(ctx).name} i1;')
            tacs.append(f'pop index {mytype.name};')
        else:
            ptype = self.parent.type(ctx)
            index = ptype.member_indices[self.selector]
            tacs.append(f'pop field {ptype.name} {index} {mytype.name};')
        return tacs

class Call(HasValue, Tacable):
    """Before the call opcode, push all arguments to the function being called
    in order. Then in executing the call opcode, the stack position below
    all of the arguments will be saved, the calling code pointer will be saved,
    and execution will jump to the function location.
    When the called function returns, all of its locals will be popped,
    all of its arguments are popped, the stack pointer is reset, the return
    value is pushed, and execution resumes at the calling point.
    The site of the return instruction will need to be examined to find all of
    the scoped blocks that need to be cleared.
    """
    def __init__(self, r):
        base, resolver = r
        self.args = [] if len(resolver[-1]) == 2 else resolver[-1][1]
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
        name = self.function.name
        if name in ctx.types:
            return ctx.types[name]
        pt = self.function.type(ctx)
        if pt is not None:
            raise TypeError('Attempt to call non-function type')
        if not isinstance(self.function, Variable):
            raise NotImplementedError('Function pointers not yet supported')
        types = [arg.type(ctx) for arg in self.args]
        function = ctx.match_function(name, types)
        if function is None:
            raise NameError(f'No function found named {name} for {types}')
        return function.type.return_type
    
    def eval(self, ctx, _):
        if not isinstance(self.function, Variable):
            raise NotImplementedError('Function pointers not yet supported')
        name = self.function.name
        if name in ctx.types:
            if len(self.args) != 1:
                raise TypeError('Only one argument should be passed to a typecast')
            tacs = self.args[0].eval(ctx, _)
            tacs.append(f'cast {self.args[0].type(ctx).name} {name};')
            return tacs
        types = [arg.type(ctx) for arg in self.args]
        function = ctx.match_function(name, types)
        if function is None:
            raise NameError(f'No function found named {name} for {types}')
        tacs = []
        return_type = function.type.return_type
        for arg, typ, target in zip(self.args, types, function.type.args_types):
            tacs += arg.eval(ctx, _)
            target_type = ctx.cast_to(typ, target)
            if target_type is None:
                raise TypeError(f'Cannot cast {typ.name} to {target.name}')
            elif target_type != typ:
                tacs.append(f'cast {typ.name} {target.name};')
        tacs.append(f'call fixed {name} {return_type.name} {" ".join([arg.name for arg in function.type.args_types])};')
        for typ in function.type.args_types:
            tacs.append(f'pop void {typ.name};')
        if return_type != ctx.types['void']:
            tacs.append(f'push returned {return_type.name};')
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
        return lt
    
    def eval(self, ctx, _):
        sources = self.targets[1:] + [self.value]
        tacs = []
        for target, source, op in reversed(tuple(zip(self.targets, sources, self.ops))):
            if op == '=':
                tacs += target.assign_to(ctx, _, source)
            else:
                binop = LeftAssocBinOp([target, [[op[0], source]]])
                tacs += target.assign_to(ctx, _, binop)
        tacs += self.targets[0].eval(ctx, _)
        return tacs

class Variable(HasValue, Tacable, Assignable):
    def __init__(self, r):
        self.name = r
    
    def is_constant(self, ctx):
        if self.name not in ctx.vars:
            if not any(map(lambda f: f[0] == self.name, ctx.funcs.keys())):
                raise KeyError(f'Variable {self.name} not defined')
            return True
        return ctx.vars[self.name].is_constant
    
    def type(self, ctx):
        if self.name not in ctx.vars:
            if not any(map(lambda f: f[0] == self.name, ctx.funcs.keys())):
                raise KeyError(f'Variable {self.name} not defined')
            return None
        return ctx.vars[self.name].type
    
    def eval(self, ctx, _):
        return [f'push variable {self.name} {self.type(ctx).name};']
    
    def assign_to(self, ctx, _, value):
        if self.is_constant(ctx):
            raise TypeError(f'Cannot assign to constant variable {self.name}')
        tacs = value.eval(ctx, _)
        mytype = self.type(ctx)
        valtype = value.type(ctx)
        if mytype != valtype:
            casttype = ctx.cast_to(valtype, mytype)
            if casttype is None:
                raise TypeError(f'Cannot implicitly cast type {valtype.name} to {mytype.name}')
            tacs.append(f'cast {valtype.name} {casttype.name};')
        tacs.append(f'pop variable {self.name} {self.type(ctx).name};')
        return tacs

class Constant(HasValue, Tacable, AOTable):
    def __init__(self, r):
        if r.startswith('"'):
            self.type_ = 'string'
            self.value = r
        elif r.endswith('j'):
            self.type_ = 'complex'
            self.value = complex(r)
        elif '.' in r:
            self.type_ = 'float'
            self.value = float(r)
        elif r in ('true', 'false'):
            self.type_ = 'bool'
            self.value = r == 'true'
        else:
            self.type_ = 'int'
            self.value = int(r)
    
    def is_constant(self, ctx):
        return True
    
    def type(self, ctx):
        if self.type_ == 'string':
            return ctx.types['i1']
        elif self.type_ == 'float':
            return ctx.types['f1']
        elif self.type_ == 'int':
            return ctx.types['i1']
        elif self.type_ == 'bool':
            return ctx.types['b1']
        elif self.type_ == 'complex':
            return ctx.types['c1']
        raise TypeError(f'Unknown constant type {self.type_}')
    
    def eval(self, ctx, _):
        if self.type_ == 'string':
            if self.value not in ctx.keys:
                ctx.keys[self.value] = len(ctx.keys)
            value = ctx.keys[self.value]
        elif self.type_ == 'bool':
            value = int(self.value)
        else:
            value = self.value
        return [f'push const {value} {self.type(ctx).name};']
    
    def const_value(self):
        if self.type_ in ('int', 'float', 'complex'):
            return self.value
        elif self.type_ == 'bool':
            return int(self.value)
        return None

class UnaryOp(HasValue, Tacable, AOTable):
    def __init__(self, r):
        if len(r) == 2:
            if len(r[0]) == 1:
                self.value = r[-1]
                self.op = r[0][0]
            else:
                self.value = UnaryOp([r[0][1:], r[1]])
                self.op = r[0][0]
        else:
            self.value = r[0]
            self.op = None
    
    def is_constant(self, ctx):
        return self.value.is_constant(ctx)
    
    def type(self, ctx):
        return self.value.type(ctx)
    
    def eval(self, ctx, _):
        tacs = self.value.eval(ctx, _)
        if self.op is not None:
            tacs.append(f'unary {self.op} {self.type(ctx).name};')
        return tacs
    
    def const_value(self):
        if not isinstance(self.value, AOTable):
            return None
        value = self.value.const_value()
        if value is None:
            return None
        if self.op == '-':
            return -value
        elif self.op == '~':
            return ~value
        elif self.op == '!':
            return not value

class Return(Tacable):
    """When a return instruction is reached in post-processing, we must
    traverse each scope upwards to the root of the function and pop each of
    their locals. Then, get the calling address, pop all of the arguments,
    push the return value, if any, and return control to the calling address.
    """
    def __init__(self, r):
        if len(r) == 1:
            self.value = None
        self.value = r[1]
    
    def eval(self, ctx, _):
        if self.value is None:
            return ['return void;']
        tacs = self.value.eval(ctx, _)
        tacs.append(f'return {self.value.type(ctx).name};')
        return tacs

class If(Tacable):
    def __init__(self, r):
        self.predicate = r[1]
        self.if_true = r[2]
        if len(r) == 4:
            self.if_false = r[3][1]
        else:
            self.if_false = None
    
    def eval(self, ctx, _):
        tacs = self.predicate.eval(ctx, _)
        true_block = self.if_true.eval(ctx, _)
        false_block = []
        if self.if_false is not None:
            false_block = self.if_false.eval(ctx, _)
            true_block.append(f'jmp back {len(false_block)} always;')
        tacs.append(f'jmp ahead {len(true_block)} if false;')
        return tacs + true_block + false_block

class For(Tacable):
    def __init__(self, r):
        self.before = r[1]
        self.predicate = r[2]
        self.after_each = r[3]
        self.statement = r[4]
    
    def eval(self, ctx, _):
        tacs = self.before.eval(ctx, _)
        condition = self.predicate.eval(ctx, _)
        body = self.statement.eval(ctx, _)
        after = self.after_each.eval(ctx, _)
        tacs.append(f'label for begin {len(condition) + len(body) + 2};')
        tacs += condition
        tacs.append(f'jmp ahead {len(body) + len(after) + 2} if false;')
        tacs += body
        tacs += after
        tacs.append(f'jmp back {len(after) + len(body) + len(condition)} always;')
        tacs.append('label for end;')
        return tacs

class While(Tacable):
    def __init__(self, r):
        self.predicate = r[1]
        self.statement = r[2]
    
    def eval(self, ctx, _):
        tacs = ['label while begin;']
        predicate = self.predicate.eval(ctx, _)
        body = self.statement.eval(ctx, _)
        tacs += predicate
        tacs.append(f'jmp ahead {len(body) + 2} if false;')
        tacs += body
        tacs.append(f'jmp back {len(body) + len(predicate)} always;')
        tacs.append('label while end;')
        return tacs

class Break:
    """As break can occur only within for or while loops, when break is
    encountered in post-processing, we iterate upwards until we find a
    begin label for a for or while loop, then iterate downward until we find
    its matching end label.
    Then, for all scoped blocks that include this statement but not the
    end label, pop all of their locals off the stack. Finally, jump execution
    to the position immediately after the end label.
    """
    def __init__(self, r):
        pass
    
    def eval(self, ctx, _):
        return ['break;']

class Continue:
    """Behaves the same as Break, but jump to the begin label instead of the
    end label.
    """
    def __init__(self, r):
        pass
    
    def eval(self, ctx, _):
        return ['continue;']

class Typespec:
    def __init__(self, r):
        if len(r[1]) == 0:
            self.base = None
            self.name = r[0]
            self.array = False
        else:
            self.base = Typespec([r[0], r[1][:-1]])
            self.name = self.base.key()
            self.array = True
    
    def key(self):
        if not self.array:
            return self.name
        return f'{self.base.key()}[]'
    
    def type(self, ctx):
        if self.base is None:
            return ctx.types[self.name]
        key = self.key()
        if key in ctx.types:
            # Array type already exists, just return it
            return ctx.types[key]
        # Array type does not exist, create it
        array_type = compiler.Type(key, {}, {}, base_type = self.base)
        ctx.types[key] = array_type
        return array_type

class ExprStmt(Tacable):
    def __init__(self, r):
        self.expr = r[0]
    
    def eval(self, ctx, _):
        tacs = self.expr.eval(ctx, _)
        """In some cases this will need to be optimized away.
        push X t;
        pop void t;
        can annihilate each other, and
        push X void;
        should never occur, though function calls that return void
        will push nothing.
        pop void void;
        will be added when the expression is void, e.g. void function calls;
        this can just be omitted always.
        """
        tacs.append(f'pop void {self.expr.type(ctx).name};')
        return tacs

class BlockStmt:
    def __init__(self, r):
        self.statements = r[0]
    
    def eval(self, ctx, _):
        return ctx.eval_block(self)
