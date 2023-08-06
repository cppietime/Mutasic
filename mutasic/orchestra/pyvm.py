"""Most basic Python-backed VM for opcodes."""

import math
import random

import numpy as np

class VM:
    def __init__(self):
        self.stack = []
        self.global_vars = []
        self.pre_init = []
        self.functions = {}
        self.retval = None
        
        self.builtin_funcs = {}
        
        self.block_size = 64
    
    def run(self):
        self.run_function(self.pre_init)
        
        main_name = 'main()'
        if main_name not in self.functions:
            raise ValueError('No main function')
        
        self.run_function(self.functions[main_name])
    
    def run_function(self, function):
        pc = 0
        returned = False
        stack_base = len(self.stack)
        while not returned:
            opcode = function[pc]
            pc, returned = self.simulate(opcode, pc)
        self.stack = self.stack[:stack_base]
    
    def simulate(self, opcode, pc):
        stripped_sc = opcode.strip()[:-1].strip()
        words = stripped_sc.split()
        if words[0] == 'noop':
            return pc + 1, True
        if words[0] == 'return':
            return pc, False
        # TODO opcodes
    
    def add(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        # TODO type-specific addition
        self.stack.append(result)
    
    def sub(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        # TODO type-specific addition
        self.stack.append(result)
    
    def mul(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        # TODO type-specific addition
        self.stack.append(result)
    
    def div(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        # TODO type-specific addition
        self.stack.append(result)
    
    def mod(self, tl, tr):
        right = self.stack.pop()
        left = self.stack.pop()
        # TODO type-specific addition
        self.stack.append(result)
    
    def push(self, words):
        if words[1] == 'variable':
            index = int(words[3])
            if words[2] == 'stack':
                # local
            elif words[2] == 'param':
                # func params
            elif words[2] == 'global':
                # global
            pass
        elif words[1] == 'const':
            pass
        elif words[1] == 'index':
            pass
        elif words[1] == 'field':
            pass
        elif words[1] == 'returned':
            pass
        self.stack.append(value)
    
    def pop(self, words):
        popped = self.stack.pop()
        if words[1] == 'variable':
            if words[2] == 'stack':
                # local
            elif words[2] == 'param':
                # func params
            elif words[2] == 'global':
                # global
            pass
        elif words[1] == 'index':
            pass
        elif words[1] == 'field':
            pass
        elif words[1] == 'void':
            pass
    
    def jmp(self, words, pc):
        direction, offset, condition = words[1:4]
        direction = -1 if direction = 'back' else 1
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
        full_funcname = f'funcname({",".join(argtypes)})'
        if full_funcname in self.builtin_funcs:
            self.builtin_funcs[full_funcname]()
            return
        if full_funcname not in self.functions:
            raise NameError(f'Function {full_funcname} not defined')
        function = self.functions[full_funcname]
        self.run_function(self, function)
    
    def advance(self, amt, types):
        for typ in types:
            if typ[-1] == '1':
                self.stack.push(0)
            elif typ[-1] == 'm':
                self.stack.push(np.zeros(self.block_size))
            else:
                raise NotImplementedError('Arrays not yet implemented')
    
    def regress(self, amt):
        self.stack = self.stack[:-amt]
