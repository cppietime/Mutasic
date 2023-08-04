"""Core elements of combinator based parsing."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import re
from typing import Callable, ClassVar

@dataclass(frozen=True)
class ParserState:
    """The immutable parser state at the core of all combinators."""
    
    """Really just needs to allow indexing/slicing and len.
    Can be buffered on disk, for example.
    """
    source: str
    
    position: int = 0
    
    line: int = 0
    column: int = 0
    
    error_reason: str = None
    error_message: str = None
    
    result: any = None
    
    def __post_init__(self):
        if self.result is not None and self.error_reason is not None:
            raise ValueError('Must not set error_reason and result in the same state')
    
    @property
    def is_error(self):
        return self.error_reason is not None
    
    def update(self, **kwargs):
        """Create and return a new state with updated values from kwargs."""
        if kwargs.get('error_reason', None) is not None:
            if kwargs.get('result', None) is not None:
                raise ValueError('Must not set error_reason and result in the same state')
            kwargs['result'] = None
        elif kwargs.get('result', None) is not None:
            kwargs['error_reason'] = None
            kwargs['error_message'] = None
            if isinstance(kwargs['result'], list):
                kwargs['result'] = list(filter(lambda x: x is not None, kwargs['result']))
        return replace(self, **kwargs)
    
    def memo_key(self):
        return (self.source, self.position)
    
    def report_error(self):
        if not self.is_error:
            return
        print(f'Error in parsing at position {self.position}')
        line_no = self.source[:self.position].count('\n')
        print(f'Line {line_no}: {self.source[self.position:self.position+10]}')
        print(f'{self.error_reason}: {self.error_message}')

ParserFunc = Callable[[ParserState], ParserState]
ResultTransformer = Callable[[any], any]
ErrorTransformer = Callable[[str, str], tuple[str, str]]
ParserSelector = Callable[[ParserState], 'Parser']

@dataclass
class Parser:
    """A parser object. A transformer from one ParserState to another.
    
    Overloads a number of operators for common combinators:
    
    a + b: concatenation of a, b.
    a * i: repeat a i times.
    a & b: match both a and b in the same position, return the match for b
        on success.
    a | b: match a if it succeeds, otherwise match b.
    a ^ b: match a if b does not match, match b if a does not match,
        otherwise fail.
    a @ b: match the longer of a or b.
    -a: match the complement of a.
    a - b: match a as long as b does not match.
    a / b: match a sequence of a zero or more times delimited by b, allowing
        a trailing delimiter.
    a // b: match a sequence of a zero or more times delimited by b, rejecting
        a trailing delimiter.
    """
    function: ParserFunc
    
    """Allow auto-memoization within context blocks. When this value > 0,
    all parsers are memoized upon creation.
    """
    _memoize_depth: ClassVar[int] = 0
    
    def __post_init__(self):
        if Parser._memoize_depth > 0:
            old_depth = Parser._memoize_depth
            Parser._memoize_depth = 0
            child = Parser(self.function)
            self << child.memoize()
            Parser._memoize_depth = old_depth
    
    def then(self, transformer: ParserFunc):
        """Create and return a new parser that executes this one
        and then transforms the returned parser state.
        """
        def parser_func(state):
            state = self.function(state)
            return transformer(state)
        return Parser(parser_func)
    
    def map(self, transformer: ResultTransformer):
        """Create and return a new parser that executes this one
        and transforms the result in some way.
        
        Errors are not mapped.
        """
        def parser_func(state):
            if state.is_error:
                return state
            return state.update(result = transformer(state.result))
        return self.then(parser_func)
    
    def map_error(self, transformer: ErrorTransformer):
        """Create and return a new parser that executes this one
        and transforms the error.
        
        Success is not mapped.
        """
        def parser_func(state):
            if not state.is_error:
                return state
            error_reason, error_message = transformer(state.error_reason, state.error_message)
            return state.update(error_reason = error_reason, error_message = error_message)
        return self.then(parser_func)
    
    def chain(self, selector: ParserSelector):
        """Creates and returns a new parser that decides its parser
        to use based on the result of this parser.
        """
        def parser_func(state):
            parser = selector(state)
            return parser.function(state)
        return self.then(parser_func)
    
    def __call__(self, source):
        state = ParserState(source)
        return self.function(state)
    
    def __add__(self, other):
        """Concatenate."""
        def parser_func(state):
            if state.is_error:
                return state
            second = other.function(state)
            if second.is_error:
                return second
            return second.update(result = [state.result, second.result])
        return self.then(parser_func)
    
    def __neg__(self):
        """Succeeds when this parser fails."""
        def parser_func(state):
            if state.is_error:
                return state.update(error_reason = None, error_message = None)
            return state.update(error_reason = 'Negate', error_message = 'Input matched negative parser')
        return self.then(parser_func)
    
    def __sub__(self, other):
        """Succeeds when this parser succeeds and other fails."""
        return (-other) & self
    
    def __or__(self, other):
        """Succeeds if either this or other succeeds.
        This has precedence over other, and will short circuit on success.
        """
        def parser_func(state):
            if state.is_error:
                return state
            base_state = state
            state = self.function(base_state)
            if state.is_error:
                return other.function(base_state)
            return state
        return Parser(parser_func)
    
    def __matmul__(self, other):
        """Chooses the longer of either this or other, if both succeeds.
        If only one succeeds, return it. Fail when both fail.
        """
        def parser_func(state):
            if state.is_error:
                return state
            me = self.function(state)
            them = other.function(state)
            if not me.is_error and them.is_error:
                return me
            elif me.is_error and not them.is_error:
                return them
            elif me.is_error and them.is_error:
                return state.update(error_reason = 'NoMatch', error_message = 'Neither of or matched')
            elif me.position >= them.position:
                return me
            return them
        return Parser(parser_func)
    
    def __and__(self, other):
        """Succeed when both this and other succeed, and return the result of
        other. Will short circuit when this fails.
        """
        def parser_func(state):
            if state.is_error:
                return state
            base_state = state
            state = self.function(base_state)
            if state.is_error:
                return state
            return other.function(base_state)
        return Parser(parser_func)
    
    def __xor__(self, other):
        """Returns this result when other fails,
        other's result when this fails,
        and failure otherwise.
        """
        def parser_func(state):
            if state.is_error:
                return state
            me = self.function(state)
            them = other.function(state)
            if me.is_error and not them.is_error:
                return them
            if them.is_error and not me.is_error:
                return me
            return state.update(error_reason = 'NoMatch', error_message = 'Exclusive OR failed')
        return Parser(parser_func)
    
    def __mul__(self, times):
        """Repeat this parser an integer number of times.
        times >= 0
        """
        assert isinstance(times, int) and times >= 0
        def parser_func(state):
            if state.is_error:
                return state
            result = []
            for _ in range(times):
                state = self.function(state)
                if state.is_error:
                    return state.update(error_reason = 'NoMatch', error_message = 'Repeated matching failed')
                result.append(state.result)
            return state.update(result = result)
        return Parser(parser_func)
    
    def __lshift__(self, other):
        """Load a new function into this parser.
        Use this for forward-declared parsers.
        """
        self.function = other.function
        return self
    
    def optional(self):
        """Return empty success when this fails, otherwise this' result."""
        def parser_func(state):
            if state.is_error:
                return state
            me = self.function(state)
            if me.is_error:
                return state.update(result = None)
            return me
        return Parser(parser_func)
    
    def some(self, omit_empty=False):
        """Match this zero or more times, and return a list of all results
        in order. When omit_empty is True, some returns None when zero matches
        are made.
        """
        def parser_func(state):
            if state.is_error:
                return state
            result = []
            while True:
                new_state = self.function(state)
                if new_state.is_error:
                    break
                if new_state.position == state.position:
                    break
                state = new_state
                result.append(state.result)
            if omit_empty and not result:
                result = None
            return state.update(result = result)
        return Parser(parser_func)
    
    def many(self):
        """Match this parser one or more times and return a list of each
        result in order.
        """
        def parser_func(state):
            if state.is_error:
                return state
            matched = False
            result = []
            while True:
                new_state = self.function(state)
                if new_state.is_error:
                    break
                if new_state.position == state.position:
                    break
                state = new_state
                result.append(state.result)
                matched = True
            if not matched:
                return state.update(error_reason = 'NoMatch', error_message = 'No matches found for many')
            return state.update(result = result)
        return Parser(parser_func)
    
    def suppress(self):
        """Set the result of this to None, so it can be elided if
        concatenated.
        """
        return self.map(lambda result: None)
    
    def between(self, left, right, suppress=False):
        """Match this parser between left and right, and return a list of
        [left, this, right]. When suppress=True, left and right are suppressed,
        but the returned value is still a list.
        """
        b = concatenate(left, self, right)
        if suppress:
            b = b.map(lambda result: result[1:-1])
        return b
    
    def separated_by(self, delim, trailing=True):
        """Create and return a parser for expressions matching this,
        separated by the delimiter/divisor.
        """
        def parser_func(state):
            if state.is_error:
                return state
            result = []
            pos_state = state
            while True:
                value_state = self.function(pos_state)
                if value_state.is_error:
                    break
                state = value_state
                result.append(state.result)
                delim_state = delim.function(state)
                if delim_state.is_error:
                    break
                pos_state = delim_state
                if trailing:
                    state = delim_state
            return state.update(result = result)
        return Parser(parser_func)
    
    def __truediv__(self, delim):
        """Return repetitions of this separated by the delimeter, allowing for
        a trailing delimeter. The delimeter is not included in the results.
        """
        return self.separated_by(delim, True)
    
    def __floordiv__(self, delim):
        """Return repetitions of this separated by the delimeter, rejecting any
        trailing delimeters. The delimeter is not included in the results.
        """
        return self.separated_by(delim, False)
    
    def memoize(self, memo = None):
        """Return a copy of this parser with an attached memo table, such that
        any subsequent calls with the same source and position will not need
        to execute this parser.
        """
        old_depth = Parser._memoize_depth
        Parser._memoize_depth = 0
        if memo is None:
            memo = {}
        def parser_func(state):
            key = (state.source, state.position)
            if key in memo:
                return memo[key]
            state = self.function(state)
            memo[key] = state
            return state
        parser = Parser(parser_func)
        Parser._memoize_depth = old_depth
        return parser
    
    def guard_left_recursion(self):
        """Fail if left-recursion on this parser is detected during execution.
        """
        guard = []
        def parser_func(state):
            if guard:
                return state.update(error_reason = 'Recursion', error_message = 'Attempted left-recursion on a guarded parser')
            guard.append(1)
            state = self.function(state)
            guard.pop()
            return state
        return Parser(parser_func)
    
    def flatten(self):
        """Transforms the result such that each list element is unpacked
        in its position, and non-lists are left as-is.
        E.g. [a, [b], [c, d, [e]]] -> [a, b, c, d, [e]]
        """
        def transformer(result):
            if not isinstance(result, list):
                return result
            flat = []
            for x in result:
                if isinstance(x, list):
                    flat += x
                else:
                    flat.append(x)
            return flat
        return self.map(transformer)
    
    def select(self, i):
        """Returns only the i-th element in the result.
        Can be used as an "unwrap" mapping.
        E.g. [a, b, c].select(1) -> b
        """
        return self.map(lambda result: result[i])
    
    def wrap(self):
        """Wrap the result in another layer of list.
        E.g. [a, [b, [c]], d] -> [[a, [b, [c]], d]]
        """
        return self.map(lambda result: [result])
    
    def extract(self):
        """If the result is a 1-list, return its element.
        If it is empty, return None.
        """
        def transformer(result):
            if not isinstance(result, list):
                return result
            if len(result) == 1:
                return result[0]
            if len(result) == 0:
                return None
            return result
        return self.map(transformer)
    
    def extract_each(self):
        """Same effect as calling extract on each child result."""
        def transformer(result):
            if not isinstance(result, list):
                return result
            lst = []
            for r in result:
                if not isinstance(r, list) or len(r) > 1:
                    lst.append(r)
                if len(r) == 1:
                    lst += r
            return lst
        return self.map(transformer)
    
    def prepend(self, value):
        """Prepend a value to a list result.
        E.g. [a, b].prepend(z) -> [z, a, b]
        """
        def transformer(result):
            if not isinstance(result, list):
                return result
            return [value] + result
        return self.map(transformer)
    
    def append(self, value):
        """Append a value to a list result.
        E.g. [a, b].append(z) -> [a, b, z]
        """
        def transformer(result):
            if not isinstance(result, list):
                return result
            return result + [value]
        return self.map(transformer)
    
    def map_at(self, transformers):
        """Apply an optional map to each value.
        transformers: a dict of {index: transformer}
        """
        def transformer(result):
            if not isinstance(result, list):
                return result
            lst = []
            for i, res in enumerate(result):
                if i in transformers:
                    lst.append(transformers[i](res))
                else:
                    lst.append(res)
            return res
        return self.map(transformer)

def concatenate(*parsers):
    """A parser that runs every parser in parser in order, and returns a list
    of each parser's result. Fails when any of parsers fail.
    """
    def parser_func(state):
        if state.is_error:
            return state
        result = []
        for parser in parsers:
            state = parser.function(state)
            if state.is_error:
                return state
            result.append(state.result)
        return state.update(result = result)
    return Parser(parser_func)

def alternate(*parsers):
    """Chooses the first match among the options."""
    def parser_func(state):
        if state.is_error:
            return state
        for parser in parsers:
            candidate_state = parser.function(state)
            if not candidate_state.is_error:
                return candidate_state
        return state.update(error_reason = 'NoMatch', error_message = 'None of the candidates for alternation matched.')
    return Parser(parser_func)

def longest(*parsers):
    """Like alternate, but chooses the longest match"""
    def parser_func(state):
        if state.is_error:
            return state
        best_state = None
        for parser in parsers:
            candidate_state = parser.function(state)
            if candidate_state.is_error:
                continue
            if best_state is None or candidate_state.position > best_state.position:
                best_state = candidate_state
        if best_state is None:
            return state.update(error_reason = 'NoMatch', error_message = 'None of the candidates for alternation matched.')
        return best_state
    return Parser(parser_func)

def forward():
    """A vacuous parser that passes the input state unchanged. Can be used for
    forward-declared parsers in recursion.
    """
    def parser_func(state):
        return state
    return Parser(parser_func)

def success(result):
    """A parser that always returns a certain success value."""
    def parser_func(state):
        return state.update(result = result)
    return Parser(parser_func)

def failure(reason, message):
    """A parser that always returns failure."""
    def parser_func(state):
        return state.update(error_reason = reason, error_message = message)
    return Parser(parser_func)

def literal(value):
    """Matches a literal string."""
    def parser_func(state):
        if state.is_error:
            return state
        if state.position + len(value) > len(state.source):
            return state.update(error_reason = 'EOF', error_message = f'Reached end of source before literal "{value}" could be matched')
        if state.source[state.position : state.position + len(value)] != value:
            return state.update(error_reason = 'NoMatch', error_message = f'Did not match literal "{value}"')
        line, col = state.line, state.column
        newlines = value.count('\n')
        if newlines:
            line += newlines
            cri = value.rfind('\n')
            col = len(value) - cri
        else:
            col += len(value)
        return state.update(position = state.position + len(value), result = value, line = line, column = col)
    return Parser(parser_func)

def regex(expr):
    """Matches a regex."""
    reg = re.compile(expr)
    def parser_func(state):
        if state.is_error:
            return state
        m = reg.match(state.source[state.position:])
        if not m:
            return state.update(error_reason = 'NoMatch', error_message = f'Did not match regex "{expr}"')
        group = m.group(0)
        line, col = state.line, state.column
        newlines = group.count('\n')
        if newlines:
            line += newlines
            cri = group.rfind('\n')
            col = len(group) - cri
        else:
            col += len(group)
        return state.update(position = state.position + len(group), result = group, line = line, column = col)
    return Parser(parser_func)

def digits():
    """Matches one or more decimal digits."""
    return regex("\\d+")

def alphanumeric():
    """Matches one or more alphanumeric characters."""
    return regex("\\w+")

def alpha():
    """Matches one or more ASCII English letters."""
    return regex("[A-Za-z]+")

def ws():
    """Matches whitespace characters."""
    return regex("\\s+")

def number():
    """Matches a positive or negative integer or float."""
    return regex("-?(0|[1-9][0-9]*(\\.[0-9]*)?|\\.[0-9]+)")

def identifier():
    """Matches a valid c-style identifier."""
    return regex("[A-Za-z_][A-Za-z0-9_]*")

def lazy(function):
    """Matches using a parser obtained by calling function at parse-time."""
    def parser_func(state):
        parser = function()
        return parser.function(state)
    return Parser(parser_func)

@contextmanager
def Memoizer():
    """Automatically memoize all Parsers created within this context manager.
    """
    Parser._memoize_depth += 1
    yield None
    Parser._memoize_depth -= 1
