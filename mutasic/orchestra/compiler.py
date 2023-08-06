from dataclasses import dataclass
import math
import numbers

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
    
    def __post_init__(self):
        self.indexed_members = list(self.fields.keys()) + list(self.methods.keys())
        self.member_indices = dict(map(lambda x: reversed(x), enumerate(self.indexed_members)))

@dataclass
class Variable:
    name: str
    type: Type
    is_constant: bool = False
    initial_val: 'ast.HasValue' = None
    rate: int = 0 # Constant
    location: tuple[str, int] = None # (reference, offset)

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
        self.pre_init = [] # To-be-generated opcodes to setup globals
        self.types: dict[str, Type] = {}
        self.num_type: Type = None
        self.keys: dict[str, int] = {}
        
        self._setup_builtin_types()
        self._setup_builtin_vars()
        self._setup_builtin_funcs()
    
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
    
    def _euler_gamma(self):
        harm = 0
        for i in range(1, 1000):
            harm += 1/i
        return harm - math.log(1, 1000)
    
    def _setup_builtin_vars(self):
        self.vars['pi'] = Variable('pi', self.types['f1'], True, math.pi)
        self.vars['tau'] = Variable('tau', self.types['f1'], True, math.pi * 2)
        self.vars['e'] = Variable('e', self.types['f1'], True, math.e)
        self.vars['gamma'] = Variable('gamma', self.types['f1'], True, self._euler_gamma())
        self.vars['phi'] = Variable('phi', self.types['f1'], True, (1 + 5**.5)/2)
        self.vars['rt2'] = Variable('rt2', self.types['f1'], True, 2**.5)
        self.vars['rthalf'] = Variable('rthalf', self.types['f1'], True, .5**.5)
        self.vars['block_size'] = Variable('block_size', self.types['i1'], True, 64)
        self.vars['sample_rate'] = Variable('sample_rate', self.types['i1'], True, 44100)
        self.vars['n_channels'] = Variable('n_channels', self.types['i1'], True, 1)
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
        self.types['f1(c1)'] = c1f = Type('f1(c1)', {}, {}, True, return_type=c1, args_types=(c1,))
        self.types['f1(f1,f1)'] = f2f = Type('f1(f1,f1)', {}, {}, True, return_type=f1, args_types=(f1, f1))
        self.types['c1(c1,f1)'] = c1f1c = Type('c1(c1,f1)', {}, {}, True, return_type=c1, args_types=(c1, f1))
        self.types['f1(i1,i1,i1)'] = read_t = Type('f1(i1,i1,i1)', {}, {}, True, return_type=f1, args_types=(i1, i1, i1))
        self.types['v(i1,i1,f1)'] = write_t = Type('v(i1,i1,f1)', {}, {}, True, return_type=None, args_types=(i1, i1, f1))
        self.funcs['sqrt', (f1.name,)] = Function('sqrt', 'sqrt', f1f) # Native
        self.funcs['sin', (f1.name,)] = Function('sin', 'sin', f1f) # Native
        self.funcs['asin', (f1.name,)] = Function('asin', 'asin', f1f) # Native
        self.funcs['cos', (f1.name,)] = Function('cos', 'cos', f1f)
        self.funcs['acos', (f1.name,)] = Function('acos', 'acos', f1f)
        self.funcs['atan2', (f1.name, f1.name)] = Function('atan2', 'atan2', f2f) # Native
        self.funcs['abs', (f1.name,)] = Function('abs', 'abs', f1f)
        self.funcs['abs', (c1.name,)] = Function('abs', 'abs', c1f)
        self.funcs['arg', (c1.name,)] = Function('arg', 'arg', c1f)
        self.funcs['log', (f1.name,)] = Function('log', 'log', f1f) # Native
        self.funcs['log', (f1.name,)] = Function('log', 'log', c1c)
        self.funcs['exp', (f1.name,)] = Function('exp', 'exp', f1f) # Native
        self.funcs['exp', (c1.name,)] = Function('exp', 'exp', c1c)
        self.funcs['pow', (f1.name, f1.name)] = Function('pow', 'pow', f2f) # Native
        self.funcs['pow', (c1.name, f1.name)] = Function('pow', 'pow', c1f1c)
        self.funcs['urand', ()] = Function('urand', 'urand', f) # Native
        self.funcs['urands', ()] = Function('urands', 'urands', fs)
        self.funcs['ceil', (f1.name,)] = Function('ceil', 'ceil', f1i)
        self.funcs['floor', (f1.name,)] = Function('floor', 'floor', f1i)
        self.funcs['min', (f1.name, f1.name)] = Function('min', 'min', f2f)
        self.funcs['max', (f1.name, f1.name)] = Function('max', 'max', f2f)
        self.funcs['readbuf', (i1.name, i1.name, i1.name)] = Function('readbuf', 'readbuf', read_t) # Native
        self.funcs['writebuf', (i1.name, i1.name, f1.name)] = Function('writebuf', 'writebuf', write_t) # Native
    
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
        
        # DEBUGGING
        # for name, func in self.funcs.items():
            # print(f'\n{name}:\n{func}')
    
    def eval_program(self, program):
        self.forward_scan_global(program)
        i = 0
        for var_name, var in self.vars.items():
            var.location = ('global', i)
            i += 1
            if var.initial_val is None:
                continue
            if isinstance(var.initial_val, numbers.Number):
                self.pre_init.append(f'push const {var.initial_val} {var.type.name};')
                self.pre_init.append(f'pop variable global {var.location[1]} {var.type.name};')
                continue
            if not var.initial_val.is_constant(self):
                raise Exception(f'Global variable {var_name} has non-constant initializer.')
            self.pre_init += var.initial_val.eval(self, self.vars)
            self.pre_init.append(f'pop variable global {var.location[1]} {var.type.name};')
        self.pre_init.append('return void;')
        for func_name, func in self.funcs.items():
            if func.definition is None:
                # Can only be true for builtin functions
                continue
            self.eval_function(func)
        # print(list(self.funcs.keys()))
        return self.funcs['main', ()].code
    
    def eval_function(self, func):
        function_scope = {}
        params = func.definition.params
        i = 0
        for param_t, param_n in params:
            if param_n in self.vars:
                raise NameError(f'Variable {param_n} shadows previously declared variable')
            param = Variable(param_n, param_t, location = ('PARAM', i))
            i += 1
            function_scope[param_n] = param
        self.vars.update(function_scope)
        tacs = self.eval_block(func.definition.body)
        tacs.append('return void;')
        stack = []
        stack_size = 0
        locations = {}
        last_loop_labels = [] # 2-tuples of (final address, stack size on entry)
        break_indices = [] # 2-tuples of (final address, current position)
        cont_indices = []
        remove_indices = []
        equivalent_location = 0
        old_locations = [-1] * len(tacs) # Replace old locations in jmps with new ones
        old_jmps = []
        # Repoint relative moves
        print('\n\nOLDTACS' + '\n'.join(tacs))
        for i, tac in enumerate(tacs):
            if tac.startswith('enter scope'):
                # stack.append(stack_size)
                stack_vars = tac[:-1].split()[2:]
                types = []
                for j, stack_var in enumerate(stack_vars):
                    name, type_ = stack_var.split('=')
                    locations[name] = stack_size + j
                    types.append(type_)
                stack.append(types)
                stack_size += len(stack_vars)
                if stack_vars:
                    tacs[i] = f'advance stack {len(stack_vars)} {" ".join(types)};'
                else:
                    remove_indices.append(i)
                    equivalent_location -= 1
            elif tac.startswith('exit scope'):
                stack_vars = tac[:-1].split()[2:]
                types = []
                for stack_var in stack_vars:
                    name, type_ = stack_var.split('=')
                    locations.pop(name)
                    types.append(type_)
                old_stack_types = stack.pop()
                if types != old_stack_types:
                    raise Exception(f'Types became corrupted between {types} and {old_stack_types}')
                old_stack_size = len(old_stack_types)
                stack_diff = old_stack_size
                if stack_diff:
                    tacs[i] = f'regress stack {stack_diff} {" ".join(reversed(types))};'
                else:
                    remove_indices.append(i)
                    equivalent_location -= 1
                stack_size = old_stack_size
            elif tac.startswith('push variable') or tac.startswith('pop variable'):
                action, _, varname, typename = tac[:-1].split()
                if varname in function_scope:
                    # Function parameter
                    location = function_scope[varname].location[1]
                    tacs[i] = f'{action} variable param {location} {typename};'
                elif varname in locations:
                    # Stack local variable
                    location = locations[varname]
                    tacs[i] = f'{action} variable stack {location} {typename};'
                elif varname in self.vars:
                    location = self.vars[varname].location
                    if location[0] != 'global':
                        raise Exception(f'{varname} was not in a valid scope')
                    # Global var
                    tacs[i] = f'{action} variable global {location[1]} {typename};'
                else:
                    raise Exception(f'{varname} was not found in any scope')
            elif tac.startswith('label for begin') or tac == 'label while begin;':
                break_indices.append([])
                cont_indices.append([])
                last_loop_labels.append((equivalent_location, len(stack), tac[:-1].split()[1:], i))
                equivalent_location -= 1
                remove_indices.append(i)
            elif tac == 'continue;':
                initial_location, old_stack_size, loop, start_i = last_loop_labels[-1]
                stack_types = []
                for j in range(len(stack) - old_stack_size):
                    stack_types += reversed(stack[-1 - j])
                stack_diff = len(stack_types)
                if stack_diff:
                    equivalent_location += 1;
                    tac = f'regress stack {stack_diff} {" ".join(stack_types)};:'
                else:
                    tac = ''
                if loop[0] == 'while':
                    offset = equivalent_location - initial_location
                    tacs[i] = f'{tac}jmp back {offset} always;'
                else:
                    # For loop...
                    contloc = int(loop[-1]) + start_i
                    cont_indices[-1].append((i, equivalent_location, contloc))
                    tacs[i] = tac
            elif tac == 'break;':
                initial_location, old_stack_size, loop, _ = last_loop_labels[-1]
                stack_types = []
                for j in range(len(stack) - old_stack_size):
                    stack_types += reversed(stack[-1 - j])
                stack_diff = len(stack_types)
                if stack_diff:
                    equivalent_location += 1;
                    tac = f'regress stack {stack_diff} {" ".join(stack_types)};:'
                else:
                    tac = ''
                tacs[i] = tac
                break_indices[-1].append((i, equivalent_location))
            elif tac == 'label for end;' or tac == 'label while end;':
                breaks = break_indices.pop()
                conts = cont_indices.pop()
                for current, final in breaks:
                    offset = equivalent_location - final
                    tacs[current] += f'jmp ahead {offset} always;'
                for current, final, ci in conts:
                    offset = old_locations[ci] - final
                    tacs[current] += f'jmp ahead {offset} always;'
                equivalent_location -= 1
                remove_indices.append(i)
                last_loop_labels.pop()
            elif tac.startswith('jmp'):
                old_jmps.append(i)
            elif tac.startswith('return'):
                new_instrs = 0
                tac = ''
                typename = tac[7:-1]
                if typename != 'void':
                    tac += f'pop return {typename};:'
                    new_instrs += 1
                if stack_size:
                    tac += f'regress stack {stack_size} {" ".join(types)};:'
                    new_instrs += 1
                tac += 'return;'
                equivalent_location += new_instrs
            elif tac.startswith('pop void'):
                if i > 0 and tacs[i - 1].startswith('push'):
                    remove_indices += [i - 1, i]
                    equivalent_location -= 2
            old_locations[i] = equivalent_location
            equivalent_location += 1
        # Repoint jumps to recalculated positions
        for i in old_jmps:
            old_tac = tacs[i]
            if old_tac.startswith('jmp ahead'):
                tail = old_tac[10:]
                old_offset = int(tail[:tail.find(' ')])
                old_i = i + old_offset
                new_i = old_locations[old_i]
                new_offset = new_i - old_locations[i] + 1
            else:
                tail = old_tac[9:]
                old_offset = int(tail[:tail.find(' ')])
                old_i = i - old_offset
                new_i = old_locations[old_i]
                new_offset = old_locations[i] - new_i + 1
            tacs[i] = tacs[i].replace(str(old_offset), str(new_offset))
        # Remove no longer used opcodes and expand them
        new_tacs = []
        for i, tac in enumerate(tacs):
            if i in remove_indices:
                continue
            new_tacs += tac.split(':')
        for param in function_scope:
            self.vars.pop(param)
        func.code = new_tacs
    
    def eval_block(self, block):
        # Keep track of locals
        block_scope = {}
        tacs = [] # List of 3-address codes
        for stmt in block.statements:
            ntac = stmt.eval(self, block_scope)
            tacs += ntac
        # Pop topmost stack frame
        local_vars = sorted(block_scope.values(), key=lambda v: (v.type.name, v.name))
        for i, var in enumerate(local_vars):
            var.location = ('local', i)
        scope_key = ' '.join(map(lambda v: f'{v.name}={v.type.name}', local_vars))
        if scope_key:
            scope_key = ' ' + scope_key
        tacs.insert(0, f'enter scope{scope_key};')
        tacs.append(f'exit scope{scope_key};')
        for local in block_scope:
            self.vars.pop(local)
        return tacs
    
    def cast_to(self, ta, tb):
        """If ta can be cast to tb, return tb. Otherwise, return None.
        """
        if ta == self.types['void'] or tb == self.types['void']:
            return None
        ta_mod = ta.name[1:]
        tb_mod = tb.name[1:]
        if ta_mod == tb_mod:
            # All elementary types can be cast if needed.
            return tb
        elif tb_mod == '1[]':
            if ta_mod == '1':
                # Single values can be cast to an array to fill it.
                return tb
            return None
        elif tb_mod == 'm':
            if ta_mod == '1':
                # Single values can be cast to fill a message block.
                return tb
            return None
        return None
    
    def _common_base(self, base1, base2):
        if base1 == base2:
            return base1
        for base in 'cfib':
            if base == base1 or base == base2:
                return base
        return None
    
    def _get_or_make_type(self, base, mod):
        if mod in '1m':
            return self.types[base + mod]
        # Arrays
        key = base + mod
        if key in self.types:
            return self.types[key]
        previous = key[:-2]
        base_type = self._get_or_make_type(base, previous)
        length = Variable(f'{key}.length', self.types['i1'], True)
        new_type = Type(key, {}, {'length': length}, base_type=base_type)
        self.types[key] = new_type
        return new_type
    
    def binop_types(self, op, tl, tr):
        """If tl op tr is a valid operation, return a 3-tuple of:
        (return type, left target cast type, right target cast type).
        Otherwise, return None.
        """
        tln, trn = tl.name, tr.name
        left_base, left_mod = tln[:1], tln[1:]
        right_base, right_mod = trn[:1], trn[1:]
        if op in ('==', '!=', '<=', '>=', '<', '>'):
            # Can compare any types of same dimension
            if left_mod != right_mod:
                return None
            base = self._common_base(left_base, right_base)
            common_type = self._get_or_make_type(base, left_mod)
            bool_type = self._get_or_make_type('b', left_mod)
            return (bool_type, common_type, common_type)
        elif op in ('||', '&&'):
            # Operands must be cast to bool before opearting in this case
            if left_mod != right_mod:
                return None
            bool_type = self._get_or_make_type('b', left_mod)
            return (bool_type, bool_type, bool_type)
        else:
            # All other binary ops should act the same
            if (left_mod, right_mod) == ('1', '1[]'):
                # Broadcast scalar to an array
                return (tr, self.types[trn[:2]], tr)
            elif (left_mod, right_mod) == ('1[]', '1'):
                # Broadcast scalar to an array
                return (tl, tl, self.types[tln[:2]])
            elif (left_mod, right_mod) == ('1', 'm'):
                # Broadcast scalar to a block
                return (tr, self.types[trn[1] + '1'], tr)
            elif (left_mod, right_mod) == ('m', '1'):
                # Broadcast scalar to a block
                return (tl, tl, self.types[tln[0] + '1'])
            elif '[]' in left_mod or '[]' in right_mod:
                # Do not support binary operations on arrays otherwise
                return None
            elif left_mod == right_mod:
                if op in ('|', '&', '^', '<<', '>>'):
                    # These ops must take and return integers
                    common_base = 'i'
                else:
                    common_base = self._common_base(left_base, right_base)
                common_type = self._get_or_make_type(common_base, left_mod)
                return (common_type, common_type, common_type)
            return None
    
    def unop_types(self, op, t):
        if op == '-':
            return (t, t)
        elif op == '~':
            # Must be integers
            itype = self._get_or_make_type('i', t.name[1:])
            return (itype, itype)
        elif op == '!':
            return (self._get_or_make_type('b', t.name[1:]), t)
        raise NameError(f'Unknown unary operator {op}')
