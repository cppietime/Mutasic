"""Core elements of combinator based parsing."""

from dataclasses import dataclass, replace
import re
from typing import Callable

@dataclass(frozen=True)
class ParserState:
    """The immutable parser state at the core of all combinators."""
    
    """Really just needs to allow indexing/slicing and len.
    Can be buffered on disk, for example.
    """
    source: str
    
    position: int = 0
    
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

ParserFunc = Callable[[ParserState], ParserState]
ResultTransformer = Callable[[any], any]
ErrorTransformer = Callable[[str, str], tuple[str, str]]
ParserSelector = Callable[[ParserState], 'Parser']

@dataclass
class Parser:
    """A parser object. A transformer from one ParserState to another."""
    function: ParserFunc
    
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
        def parser_func(state):
            if state.is_error:
                return state
            second = other.function(state)
            return second.update(result = [state.result, second.result])
        return self.then(parser_func)
    
    def __neg__(self):
        def parser_func(state):
            if state.is_error:
                return state.update(error_reason = None, error_message = None)
            return state.update(error_reason = 'Negate', error_message = 'Input matched negative parser')
        return self.then(parser_func)
    
    def __sub__(self, other):
        return self & (-other)
    
    def __or__(self, other):
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
        def parser_func(state):
            if state.is_error:
                return state
            base_state = state
            state = other.function(base_state)
            if state.is_error:
                return state
            return self.function(base_state)
        return Parser(parser_func)
    
    def __xor__(self, other):
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
    
    def __lshift__(self, function):
        """Load a new function into this parser.
        Use this for forward-declared parsers.
        """
        self.function = function
        return self
    
    def optional(self):
        def parser_func(state):
            if state.is_error:
                return state
            me = self.function(state)
            if me.iserror:
                return state
            return me
        return Parser(parser_func)
    
    def some(self):
        def parser_func(state):
            if state.is_error:
                return state
            result = []
            while True:
                new_state = self.function(state)
                if new_state.is_error:
                    break
                state = new_state
                result.append(state.result)
            return state.update(result = result)
        return Parser(parser_func)
    
    def many(self):
        def parser_func(state):
            if state.is_error:
                return state
            matched = False
            result = []
            while True:
                new_state = self.function(state)
                if new_state.is_error:
                    break
                state = new_state
                result.append(state.result)
                matched = True
            if not matched:
                return state.update(error_reason = 'NoMatch', error_message = 'No matches found for many')
            return state.update(result = result)
        return Parser(parser_func)
    
    def suppress(self):
        return self.map(lambda result: None)
    
    def between(self, left, right):
        return concatenate(left, self, right)
    
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
        return self.separated_by(delim, True)
    
    def __floordiv__(self, delim):
        return self.separated_by(delim, False)

def concatenate(*parsers):
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
    def parser_func(state):
        return state
    return Parser(parser_func)

def success(result):
    def parser_func(state):
        return state.update(result = result)
    return Parser(parser_func)

def failure(reason, message):
    def parser_func(state):
        return state.update(error_reason = reason, error_message = message)
    return Parser(parser_func)

def literal(value):
    def parser_func(state):
        if state.is_error:
            return state
        if state.position + len(value) > len(state.source):
            return state.update(error_reason = 'EOF', error_message = f'Reached end of source before literal "{value}" could be matched')
        if state.source[state.position : state.position + len(value)] != value:
            return state.update(error_reason = 'NoMatch', error_message = f'Did not match literal "{value}"')
        return state.update(position = state.position + len(value), result = value)
    return Parser(parser_func)

def regex(expr):
    reg = re.compile(expr)
    def parser_func(state):
        if state.is_error:
            return state
        m = reg.match(state.source[state.position:])
        if not m:
            return state.update(error_reason = 'NoMatch', error_message = f'Did not match regex "{expr}"')
        group = m.group(0)
        return state.update(position = state.position + len(group), result = group)
    return Parser(parser_func)

def digits():
    return regex("\\d+")

def alphanumeric():
    return regex("\\w+")

def alpha():
    return regex("[A-Za-z]+")

def ws():
    return regex("\\s+")

def number():
    """Matches a positive or negative integer or float."""
    return regex("-?(0|[1-9][0-9]*(\\.[0-9]*)?|\\.[0-9]+)")

def identifer():
    return regex("[A-Za-z_][A-Za-z0-9]*")

def lazy(function):
    def parser_func(state):
        parser = function()
        return parser.function(state)
    return Parser(parser_func)
