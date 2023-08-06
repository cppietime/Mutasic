"""Most basic Python-backed VM for opcodes."""

import math
import random

import numpy as np

class VM:
    def __init__(self):
        self.stack = []
        self.stack_base = 0
        self.global_vars = []
        self.pre_init = []
        self.functions = {}
        self.retval = None
        
        self.builtin_funcs = {
            'sin(f1)': self.sin1,
        }
        
        self.block_size = 64
    
    def run(self):
        self.run_function(self.pre_init)
        
        main_name = 'main()'
        if main_name not in self.functions:
            raise ValueError(f'No main function in {list(self.functions.keys())}')
        
        self.run_function(self.functions[main_name])
    
    def run_function(self, function):
        pc = 0
        returned = False
        old_base = self.stack_base
        self.stack_base = len(self.stack)
        while not returned:
            opcode = function[pc]
            try:
                pc, returned = self.simulate(opcode, pc)
            except Exception as e:
                raise Exception(f'Error at {pc=} for {opcode=}:\n{e}')
        self.stack = self.stack[:self.stack_base]
        self.stack_base = old_base
    
    def simulate(self, opcode, pc):
        stripped_sc = opcode.strip()[:-1].strip()
        words = stripped_sc.split()
        if words[0] == 'noop':
            pass
        elif words[0] == 'call':
            self.call(words)
        elif words[0] == 'return':
            if words[1] != 'void':
                self.retval = self.stack.pop()
            return pc, True
        elif words[0] == 'push':
            self.push(words)
        elif words[0] == 'pop':
            self.pop(words)
        elif words[0] == 'jmp':
            return self.jmp(words, pc), False
        elif words[0] == 'advance':
            self.advance(int(words[2]), words[3:])
        elif words[0] == 'regress':
            self.regress(int(words[2]))
        elif words[0] == 'unary':
            op = words[1]
            if op == '!':
                self.unary_not(words[2])
            elif op == '~':
                self.unary_inv(words[2])
            elif op == '-':
                self.unary_minus(words[2])
            else:
                raise ValueError(f'Unknown unary operator {op}')
        elif words[0] == 'binary':
            op = words[1]
            funcs = {
                '+':  self.add,
                '-':  self.sub,
                '*':  self.mul,
                '/':  self.div,
                '%':  self.mod,
                '&':  self.binand,
                '|':  self.binor,
                '^':  self.binxor,
                '==': self.eq,
                '!=': self.neq,
                '>=': self.gte,
                '<=': self.lte,
                '>':  self.gt,
                '<':  self.lt,
                '<<': self.lshift,
                '>>': self.rshift,
                '&&': self.logand,
                '||': self.logor,
            }
            func = funcs[op]
            func(words[2], words[3])
        elif words[0] == 'cast':
            from_, to = words[1:]
            operand = self.stack.pop()
            casted = self.cast_to(operand, from_, to)
            self.stack.append(casted)
        elif words[0] == 'dup':
            self.stack.append(self.stack[-1])
        elif words[0] == 'swap':
            index = int(words[1])
            tmp = self.stack[-1]
            self.stack[-1] = self.stack[-1 - index]
            self.stack[-1 - index] = tmp
        # TODO opcodes
        else:
            raise NameError(f'Unknown opcode {opcode}')
        return pc + 1, False
    
    def push(self, words):
        if words[1] == 'variable':
            index = int(words[3])
            if words[2] == 'stack':
                # local
                value = self.stack[self.stack_base + index]
            elif words[2] == 'param':
                # func params
                value = self.stack[self.stack_base - 1 - index]
            elif words[2] == 'global':
                # global
                value = self.global_vars[index]
            else:
                raise NameError(f'Unknown variable specifier {words[2]}')
        elif words[1] == 'const':
            typ = words[3]
            if typ == 'c1':
                value = complex(words[2])
            else:
                value = float(words[2])
        elif words[1] == 'index':
            index = int(self.stack.pop())
            parent = self.stack.pop()
            if not hasattr(parent, '__getitem__'):
                raise TypeError(f'Attempt to index unindexable type {type(parent)}.')
            value = parent[index]
        elif words[1] == 'field':
            typ = words[2]
            index = int(words[3])
            parent = self.stack.pop()
            if isinstance(parent, list):
                if index == 0: # Length
                    value = len(parent)
            elif typ == 'c1':
                if index == 0: # Real
                    value = parent.real
                elif index == 1:
                    value = parent.imag
            else:
                raise TypeError("Custom classes don't even exist yet")
        elif words[1] == 'head':
            index = int(words[2])
            value = self.stack[-index]
        elif words[1] == 'returned':
            value = self.retval
            if value is None:
                raise TypeError('Cannot push returned value when there is none')
            self.retval = None
        else:
            raise NameError(f'Unknown push specifier {words[1]}')
        self.stack.append(value)
    
    def pop(self, words):
        popped = self.stack.pop()
        if words[1] == 'variable':
            index = int(words[3])
            if words[2] == 'stack':
                # local
                self.stack[self.stack_base + index] = popped
            elif words[2] == 'param':
                # func params
                self.stack[self.stack_base - 1- index] = popped
            elif words[2] == 'global':
                # global
                self.global_vars[index] = popped
            else:
                raise NameError(f'Unknown variable specifier {words[2]}')
        elif words[1] == 'index':
            print('RUNNING POP!!!!')
            index = int(popped)
            value = self.stack.pop()
            parent = self.stack.pop()
            if not hasattr(parent, '__getitem__'):
                raise TypeError(f'Attempt to index unindexable type {type(parent)}.\nGot {popped=}, {value=}, {parent=},\nremainder: {self.stack=}')
            parent[index] = value
        elif words[1] == 'field':
            raise TypeError('Classes not yet implemented, how did this happen?')
        elif words[1] == 'void':
            # Do nothing
            pass
        elif words[1] == 'head':
            index = int(words[2])
            self.stack[-index] = popped
        else:
            raise NameError(f'Unknown push specifier {words[1]}')
    
    def jmp(self, words, pc):
        direction, offset, condition = words[1:4]
        direction = -1 if direction == 'back' else 1
        offset = int(offset) * direction
        if condition == 'always':
            return pc + offset
        else:
            value = words[4] == 'true'
            sval = self.stack.pop()
            if bool(sval) == value:
                return pc + offset
        return pc + 1
    
    def call(self, words):
        if words[1] != 'fixed':
            raise NotImplementedError('Non-fixed functions not yet supported')
        funcname = words[2]
        rettype = words[3]
        argtypes = words[4:]
        full_funcname = f'{funcname}({",".join(argtypes)})'
        if full_funcname in self.builtin_funcs:
            self.builtin_funcs[full_funcname]()
            return
        if full_funcname not in self.functions:
            raise NameError(f'Function {full_funcname} not defined')
        function = self.functions[full_funcname]
        self.run_function(function)
    
    def advance(self, amt, types):
        for typ in types:
            if typ[-1] == '1':
                self.stack.append(0)
            elif typ[-1] == 'm':
                self.stack.append(np.zeros(self.block_size))
            elif typ[1:] == '1[]':
                self.stack.append([])
            else:
                raise NotImplementedError(f'Type {typ} failed. Arrays not yet implemented')
    
    def regress(self, amt):
        self.stack = self.stack[:-amt]
    
    def base_type_of(self, base):
        return {
            'b': bool,
            'i': int,
            'f': float,
            'c': complex,
        }[base]
    
    def cast_to(self, value, from_, to):
        if from_ == to:
            return value
        fb, fm = from_[0], from_[1:]
        tb, tm = to[0], to[1:]
        to_base = self.base_type_of(tb)
        # Block to block
        if fm == tm == 'm':
            return value.astype(to_base)
        # Scalar to scalar
        elif fm == tm == '1':
            return to_base(value)
        # Scalar to block
        elif (fm, tm) == ('1', 'm'):
            return np.full(self.block_size, value, dtype=to_base)
        # List to list
        elif fm == tm == '1[]':
            return [to_base(v) for v in value]
        elif (fm, tm) == ('1', '1[]'):
            return [to_base(0)] * value
        raise TypeError(f'No cast from {from_} to {to}')
    
    def add(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left + right
        elif lmod == rmod == 'm':
            result = left + right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left + right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [r + left for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l + right for l in left]
        elif lmod == rmod == '1[]':
            result = [l + r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot add types {tl} and {tr}')
        self.stack.append(result)
    
    def sub(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left - right
        elif lmod == rmod == 'm':
            result = left - right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left - right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left - r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l - right for l in left]
        elif lmod == rmod == '1[]':
            result = [l - r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot subtract types {tl} and {tr}')
        self.stack.append(result)
    
    def mul(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left * right
        elif lmod == rmod == 'm':
            result = left * right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left * right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [r * left for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l * right for l in left]
        elif lmod == rmod == '1[]':
            result = [l * r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot multiply types {tl} and {tr}')
        self.stack.append(result)
    
    def div(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left / right
        elif lmod == rmod == 'm':
            result = left / right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left / right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left / r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l / right for l in left]
        elif lmod == rmod == '1[]':
            result = [l / r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot divide types {tl} and {tr}')
        self.stack.append(result)
    
    def mod(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left % right
        elif lmod == rmod == 'm':
            result = left % right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left % right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left % r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l % right for l in left]
        elif lmod == rmod == '1[]':
            result = [l % r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot modulo types {tl} and {tr}')
        self.stack.append(result)
    
    def binand(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left & right
        elif lmod == rmod == 'm':
            result = left & right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left & right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left & r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l & right for l in left]
        elif lmod == rmod == '1[]':
            result = [l & r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot binary-and types {tl} and {tr}')
        self.stack.append(result)
    
    def binor(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left | right
        elif lmod == rmod == 'm':
            result = left | right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left | right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left | r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l | right for l in left]
        elif lmod == rmod == '1[]':
            result = [l | r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot binary-or types {tl} and {tr}')
        self.stack.append(result)
    
    def binxor(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left ^ right
        elif lmod == rmod == 'm':
            result = left ^ right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left ^ right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left & r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l ^ right for l in left]
        elif lmod == rmod == '1[]':
            result = [l ^ r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot binary-xor types {tl} and {tr}')
        self.stack.append(result)
    
    def lshift(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left << right
        elif lmod == rmod == 'm':
            result = left << right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left << right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left << r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l << right for l in left]
        elif lmod == rmod == '1[]':
            result = [l << r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot bitshift types {tl} and {tr}')
        self.stack.append(result)
    
    def rshift(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left >> right
        elif lmod == rmod == 'm':
            result = left >> right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left >> right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left >> r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l >> right for l in left]
        elif lmod == rmod == '1[]':
            result = [l >> r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot bitshift types {tl} and {tr}')
        self.stack.append(result)
    
    def logand(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = bool(left and right)
        elif lmod == rmod == 'm':
            result = (left & right).astype(bool)
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = (left & right).astype(bool)
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left and r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l and right for l in left]
        elif lmod == rmod == '1[]':
            result = [l and r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot logical-and types {tl} and {tr}')
        self.stack.append(result)
    
    def logor(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = bool(left or right)
        elif lmod == rmod == 'm':
            result = (left | right).astype(bool)
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = (left | right).astype(bool)
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left or r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l or right for l in left]
        elif lmod == rmod == '1[]':
            result = [l or r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot logical-or types {tl} and {tr}')
        self.stack.append(result)
    
    def eq(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left == right
        elif lmod == rmod == 'm':
            result = left == right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left == right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left == r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l == right for l in left]
        elif lmod == rmod == '1[]':
            result = [l == r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot compare types {tl} and {tr}')
        self.stack.append(result)
    
    def neq(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left != right
        elif lmod == rmod == 'm':
            result = left != right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left != right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left != r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l != right for l in left]
        elif lmod == rmod == '1[]':
            result = [l != r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot compare types {tl} and {tr}')
        self.stack.append(result)
    
    def gte(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left >= right
        elif lmod == rmod == 'm':
            result = left >= right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left >= right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left >= r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l >= right for l in left]
        elif lmod == rmod == '1[]':
            result = [l >= r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot compare types {tl} and {tr}')
        self.stack.append(result)
    
    def lte(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left <= right
        elif lmod == rmod == 'm':
            result = left <= right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left <= right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left <= r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l <= right for l in left]
        elif lmod == rmod == '1[]':
            result = [l <= r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot compare types {tl} and {tr}')
        self.stack.append(result)
    
    def gt(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left > right
        elif lmod == rmod == 'm':
            result = left > right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left > right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left > r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l > right for l in left]
        elif lmod == rmod == '1[]':
            result = [l > r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot compare types {tl} and {tr}')
        self.stack.append(result)
    
    def lt(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        lmod, rmod = tl[1:], tr[1:]
        if lmod == rmod == '1':
            result = left < right
        elif lmod == rmod == 'm':
            result = left < right
        elif (lmod, rmod) in ( ('1', 'm'), ('m', '1') ):
            result = left < right
        elif (lmod, rmod) == ('1', '1[]'):
            result = [left < r for r in right]
        elif (lmod, rmod) == ('1[]', '1'):
            result = [l < right for l in left]
        elif lmod == rmod == '1[]':
            result = [l < r for l, r in zip(left, right, strict=True)]
        else:
            raise TypeError(f'Cannot compare types {tl} and {tr}')
        self.stack.append(result)
    
    def unary_minus(self, t):
        operand = self.stack.pop()
        tmod = t[1:]
        if tmod == '1[]':
            value = [-x for x in operand]
        else:
            value = -operand
        self.stack.append(value)
    
    def unary_not(self, t):
        operand = self.stack.pop()
        tmod = t[1:]
        if tmod == '1[]':
            value = [not x for x in operand]
        elif tmod == '~':
            value = ~operand
        else:
            value = not operand
        self.stack.append(value)
    
    def unary_inv(self, t):
        operand = self.stack.pop()
        tmod = t[1:]
        if tmod == '1[]':
            value = [~x for x in operand]
        else:
            value = ~operand
        self.stack.append(value)
    
    def sin1(self):
        phase = self.stack[-1]
        self.retval = math.sin(phase)
