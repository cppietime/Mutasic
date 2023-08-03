"""Token interface for use with lexers and parser."""

from abc import ABC
from dataclasses import dataclass

class Token(ABC):
    """Abstract class to derive for tokens."""
    def token_name(self):
        return NotImplemented
    
    def token_id(self):
        return NotImplemented
    
    def token_lexeme(self):
        return NotImplemented
    
    def token_lineno(self):
        return NotImplemented
    
    def token_colno(self):
        return NotImplemented
    
    def token_file(self):
        return NotImplemented
    
    def __str__(self):
        return f'{self.token_name()}("{self.token_lexeme()}")'
    
    def __repr__(self):
        return str(self)

class NamedToken(Token):
    def __init__(self, name):
        self.name = name
    
    def token_name(self):
        return self.name

@dataclass
class StaticToken(Token):
    """Out of the box implementation of a token with all properties known
    at init time.
    """
    name: str
    lexeme: str
    line_no: int = 0
    col_no: int = 0
    file: str = ''
    id_: int = 0
    
    def token_name(self):
        return self.name
    
    def token_id(self):
        return self.id_
    
    def token_lexeme(self):
        return self.lexeme
    
    def token_lineno(self):
        return self.line_no
    
    def token_colno(self):
        return self.col_no
    
    def token_file(self):
        return self.file
