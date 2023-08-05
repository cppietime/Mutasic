from dataclasses import dataclass

from . import ast

"""We have the following types:
i1, im,
f1, fm,
b1, bm,
c1, cm,

upgrade types: b->i->f->c

Builtins:
block_size: i1
sample_rate: i1
n_channels: i1
glob_time: im
active_notes: i1
note_num: i1
note_octlets: i1
note_amp: f1
note_extra_num: i1
note_extra: i1[]
note_voice: i1
note_time: im
note_dead_time: im
output: fm
keep_alive: i1
"""

FunctionSig = tuple[str, tuple['Type', ...]]
Scope = dict[str, 'Variable']

@dataclass
class Type:
    name: str
    methods: dict[FunctionSig, 'Function']
    fields: dict[str, 'Variable']
    is_function: bool = False
    base_type: 'Type' = None
    return_type: 'Type' = None
    args_types: list['Type'] = None

@dataclass
class Variable:
    name: str
    type: Type
    is_constant: bool = False
    initial_val: 'ast.HasValue' = None
    rate: int = 0 # Constant

@dataclass
class Function:
    full_name: str
    base_name: str
    type: Type
    definition: 'ast.Funcdef' = None
    code: list = None

class Context:
    def __init__(self):
        self.vars: Scope = {}
        self.funcs: dict[FunctionSig, Function] = {}
        self.types: dict[str, Type] = {}
        self.num_type: Type = None
        self.keys: dict[str, int] = {}
        
        self._setup_builtin_types()
        self._setup_builtin_vars()
        self._setup_builtin_funcs()
        
        self.control_stack = []
    
    def _setup_builtin_types(self):
        void = self.types['void'] = Type('void', {}, {})
        i1 = self.types['i1'] = Type('i1', {}, {})
        im = self.types['im'] = Type('im', {}, {}, base_type=i1)
        i1_len = Variable('i1[].length', i1, True)
        iarr = self.types['i1[]'] = Type('i1[]', {}, {'length': i1_len}, base_type=i1)
        f1 = self.types['f1'] = Type('f1', {}, {})
        fm = self.types['fm'] = Type('fm', {}, {}, base_type=f1)
        f1_len = Variable('f1[].length', i1, True)
        farr = self.types['f1[]'] = Type('f1[]', {}, {'length': f1_len}, base_type=f1)
        b1 = self.types['b1'] = Type('b1', {}, {})
        bm = self.types['bm'] = Type('bm', {}, {}, base_type=b1)
        b1_len = Variable('b1[].length', i1, True)
        barr = self.types['b1[]'] = Type('b1[]', {}, {'length': b1_len}, base_type=b1)
        c1 = self.types['c1'] = Type('c1', {}, {
            'real': Variable('c1.real', f1, True),
            'imag': Variable('c1.imag', f1, True),
        })
        cm = self.types['cm'] = Type('cm', {}, {}, base_type=c1)
        carr = self.types['c1[]'] = Type('c1[]', {}, {
            'length': Variable('c1[].length', i1, True),
        }, base_type=c1)
        
        self.num_type = f1
    
    def _setup_builtin_vars(self):
        self.vars['block_size'] = Variable('block_size', self.types['i1'], True) # constant
        self.vars['sample_rate'] = Variable('sample_rate', self.types['i1'], True) # constant
        self.vars['n_channels'] = Variable('n_channels', self.types['i1'], True) # constant
        self.vars['glob_time'] = Variable('glob_time', self.types['im'], True, rate=2)
        self.vars['active_notes'] = Variable('active_notes', self.types['i1'], True, rate=2)
        self.vars['note_num'] = Variable('note_num', self.types['i1'], True, rate=1)
        self.vars['note_octlets'] = Variable('note_octlets', self.types['i1'], True, rate=1)
        self.vars['note_amp'] = Variable('note_amp', self.types['f1'], True, rate=1)
        self.vars['note_extra_num'] = Variable('note_extra_num', self.types['i1'], True, rate=1)
        self.vars['note_extra'] = Variable('note_extra', self.types['i1[]'], True, rate=1)
        self.vars['note_voice'] = Variable('note_voice', self.types['i1'], True, rate=1)
        self.vars['note_time'] = Variable('note_time', self.types['im'], True, rate=2)
        self.vars['note_dead_time'] = Variable('note_dead_time', self.types['im'], True, rate=2)
        self.vars['output'] = Variable('output', self.types['fm'], False, rate=2)
        self.vars['keep_alive'] = Variable('keep_alive', self.types['i1'], False, rate=2)
    
    def _setup_builtin_funcs(self):
        f1 = self.types['f1']
        fm = self.types['fm']
        farr = self.types['f1[]']
        c1 = self.types['c1']
        i1 = self.types['i1']
        self.types['f1()'] = f = Type('f1()', {}, {}, True, return_type=f1, args_types=())
        self.types['fm()'] = fs = Type('fm()', {}, {}, True, return_type=fm, args_types=())
        self.types['f1(f1)'] = f1f = Type('f1(f1)', {}, {}, True, return_type=f1, args_types=(f1,))
        self.types['i1(f1)'] = f1i = Type('i1(f1)', {}, {}, True, return_type=i1, args_types=(f1,))
        self.types['c1(c1)'] = c1c = Type('c1(c1)', {}, {}, True, return_type=c1, args_types=(c1,))
        self.types['f1(f1,f1)'] = f2f = Type('f1(f1,f1)', {}, {}, True, return_type=f1, args_types=(f1, f1))
        self.types['c1(c1,f1)'] = c1f1c = Type('c1(c1,f1)', {}, {}, True, return_type=c1, args_types=(c1, f1))
        self.types['f1(i1,i1,i1)'] = read_t = Type('f1(i1,i1,i1)', {}, {}, True, return_type=f1, args_types=(i1, i1, i1))
        self.types['v(i1,i1,f1)'] = write_t = Type('v(i1,i1,f1)', {}, {}, True, return_type=None, args_types=(i1, i1, f1))
        self.funcs['sin', (f1.name,)] = Function('sin', 'sin', f1f)
        self.funcs['cos', (f1.name,)] = Function('cos', 'cos', f1f)
        self.funcs['abs', (f1.name,)] = Function('abs', 'abs', f1f)
        self.funcs['abs', (c1.name,)] = Function('abs', 'abs', c1c)
        self.funcs['log', (f1.name,)] = Function('log', 'log', f1f)
        self.funcs['exp', (f1.name,)] = Function('exp', 'exp', f1f)
        self.funcs['exp', (c1.name,)] = Function('exp', 'exp', c1c)
        self.funcs['pow', (f1.name, f1.name)] = Function('pow', 'pow', f2f)
        self.funcs['pow', (c1.name, f1.name)] = Function('pow', 'pow', c1f1c)
        self.funcs['urand', ()] = Function('urand', 'urand', f)
        self.funcs['urands', ()] = Function('urands', 'urands', fs)
        self.funcs['int', (f1.name,)] = Function('int', 'int', f1i)
        self.funcs['ceil', (f1.name,)] = Function('ceil', 'ceil', f1i)
        self.funcs['floor', (f1.name,)] = Function('floor', 'floor', f1i)
        self.funcs['min', (f1.name, f1.name)] = Function('min', 'min', f2f)
        self.funcs['max', (f1.name, f1.name)] = Function('max', 'max', f2f)
        self.funcs['readbuf', (i1.name, i1.name, i1.name)] = Function('readbuf', 'readbuf', read_t)
        self.funcs['writebuf', (i1.name, i1.name, f1.name)] = Function('writebuf', 'writebuf', write_t)
    
    def match_function(self, name, types):
        best = None
        for key, value in self.funcs.items():
            if key[0] != name:
                continue
            if len(key[1]) != len(types):
                continue
            fit = 2
            for p, t in zip(key[1], types):
                if self.cast_to(t, self.types[p]) == None:
                    fit = 0
                    break
                elif p != t.name:
                    fit = min(fit, 1)
            if fit == 0:
                continue
            elif fit == 1:
                if best is None:
                    best = value
            else:
                return value
        return best
    
    def forward_scan_global(self, program):
        for tl in program:
            tl.scan_scope(self, self.vars)
    
    def eval_program(self, program):
        self.forward_scan_global(program)
        for var_name, var in self.vars.items():
            if var.initial_val is None:
                continue
            if not var.initial_val.is_constant(self):
                raise Exception(f'Global variable {var_name} has non-constant initializer.')
            # TODO generate instructions to initialize global variables.
        for func_name, func in self.funcs.items():
            if func.definition is None:
                # Can only be true for builtin functions
                continue
            function_scope = {}
            params = func.definition.params
            for param_t, param_n in params:
                if param_n in self.vars:
                    raise NameError(f'Variable {param_n} shadows previously declared variable')
                param = Variable(param_n, param_t)
                function_scope[param_n] = param
            self.vars.update(function_scope)
            tacs = self.eval_block(func.definition.body)
            for param in function_scope:
                self.vars.pop(param)
            func.code = tacs
        print(list(self.funcs.keys()))
        return self.funcs['main', ('i1', 'i1')].code
    
    def eval_block(self, block):
        # Keep track of locals
        self.control_stack.append([])
        block_scope = {}
        tacs = [] # List of 3-address codes
        for stmt in block.statements:
            ntac = stmt.eval(self, block_scope)
            tacs += ntac
        # Pop topmost stack frame
        local_vars = sorted(block_scope.values(), key=lambda v: (v.type.name, v.name))
        scope_key = ' '.join(map(lambda v: f'{v.name}={v.type.name}', local_vars))
        tacs.insert(0, f'enter scope {scope_key};')
        tacs.append(f'exit_scope {scope_key};')
        for local in block_scope:
            self.vars.pop(local)
        # TODO replacements for break/continue
        self.control_stack.pop()
        return tacs
    
    def common_type(self, ta, tb):
        if ta == self.types['void'] or tb == self.types['void']:
            return None
        if ta == tb:
            return ta
        ta_base = ta.name[0]
        tb_base = tb.name[0]
        ta_mod = ta.name[1:]
        tb_mod = tb.name[1:]
        if ta_base == tb_base:
            target_base = ta_base
        else:
            for base in 'cfib':
                if ta_base == base or tb_base == base:
                    target_base = base
                    break
            else:
                return None
        if ta_mod == tb_mod:
            target_mod = ta_mod
        else:
            if ta_mod == '1' and tb_mod == 'm':
                target_mod = 'm'
            elif tb_mod == '1' and ta_mod == 'm':
                target_mod = 'm'
            else:
                return None
        return self.types[target_base + target_mod]
    
    def cast_to(self, ta, tb):
        if ta == self.types['void'] or tb == self.types['void']:
            return None
        ta_mod = ta.name[1:]
        tb_mod = tb.name[1:]
        if ta_mod == tb_mod:
            return tb
        if tb_mod == '[]':
            return None
        if tb_mod == '1':
            return None
        return tb
