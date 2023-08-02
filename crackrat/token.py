from abc import ABC
from dataclasses import dataclass

class Token(ABC):
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
    name: str
    id_: int
    lexeme: str
    line_no: int = 0
    col_no: int = 0
    file: str = ''
    
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
