"""Parse orchestra code"""

import spyceballs as sb

from .ast import *
from . import compiler, pyvm

"""Use the grammar:
PROGRAM = { FUNCDEF | VARDEC };
FUNCDEF = TYPE, id, lparen, [ PARAMS ], rparen, BLOCK;
TYPE = id, [ lbrack, rbrack ];
PARAMS = TYPE, id, { comma, TYPE, id };
BLOCK = lbrace, { STMT }, rbrace;
STMT = EXPR, semicolon | IF | WHILE | FOR | BREAK | CONTINUE | RETURN | BLOCK | VARDEC;
EXPR = { LVALUE, assign }, RVALUE;
LVALUE = VALUE, { RESOLVE };
RVALUE = COMP, { logicop, COMP };
COMP = BITS, { compop, BITS };
BITS = SHIFT, { bitop, SHIFT };
SHIFT = SUM, { shiftop, SUM };
SUM = TERM, { addop, TERM };
TERM = FACTOR, { mulop, FACTOR };
FACTOR = { unaryop }, LVALUE;
VALUE = LITERAL | id | lparen, EXPR, rparen;
RESOLVE = period, id | lbrack, EXPR, rbrack | lparen, [ ARGS ], rparen;
ARGS = EXPR, { comma, EXPR };
IF = if, lparen, EXPR, rparen, STMT, [ else, STMT ];
WHILE = while, lparen, EXPR, rparen, STMT;
FOR = for, lparen, STMT, EXPR, semicolon, STMT, rparen, STMT;
BREAK = break, semicolon;
CONTINUE = continue, semicolon;
RETURN = return, [ EXPR ], semicolon;
LITERAL = number;
VARDEC = TYPE, VAR, { comma, VAR }, semicolon;
VAR = id, [ assign, EXPR ];
"""

def gen_parser():
    expr = sb.forward()
    stmt = sb.forward()
    with sb.Memoizer():
        comment = sb.regex('\\/\\*([^*]|\n|\*[^/])*\\*\\/')
        ignore = sb.alternate(sb.ws(), comment).some()
        def tok(parser):
            return (parser + ignore).select(0)
        semicolon = tok(sb.literal(';')).suppress()
        lpar = tok(sb.literal('('))
        rpar = tok(sb.literal(')'))
        comma = tok(sb.literal(','))
        id_ = tok(sb.identifier())
        lbrack = tok(sb.literal('['))
        rbrack = tok(sb.literal(']'))
        litval = tok(sb.alternate((sb.number() + sb.literal('j')).map(lambda result: result[0] + result[1]), sb.number(), sb.regex(r'["]([^"\\]|\\.)*["]'), sb.literal('false'), sb.literal('true'))).map(Constant)
        unaryop = tok(sb.alternate(*map(sb.literal, ['-', '!', '~'])))
        addop = tok(sb.alternate(*map(sb.literal, ['+', '-'])))
        mulop = tok(sb.alternate(*map(sb.literal, ['*', '/', '%'])))
        shiftop = tok(sb.alternate(*map(sb.literal, ['<<', '>>'])))
        bitop = tok(sb.alternate(*map(sb.literal, ['|', '&', '^'])))
        compop = tok(sb.alternate(*map(sb.literal, ['==', '!=', '<=', '>=', '<', '>'])))
        logicop = tok(sb.alternate(*map(sb.literal, ['||', '&&'])))
        assign = tok(sb.alternate(*map(sb.literal, ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='])))
        return_ = sb.concatenate(tok(sb.literal('return')), expr.optional(), semicolon).map(Return)
        break_ = (tok(sb.literal('break')) + semicolon).map(Break)
        continue_ = (tok(sb.literal('continue')) + semicolon).map(Continue)
        for_ = sb.concatenate(tok(sb.literal('for')), lpar.suppress(), stmt, expr, semicolon, stmt, rpar.suppress(), stmt).map(For)
        while_ = sb.concatenate(tok(sb.literal('while')), lpar.suppress(), expr, rpar.suppress(), stmt).map(While)
        if_ = sb.concatenate(tok(sb.literal('if')), lpar.suppress(), expr, rpar.suppress(), stmt, (tok(sb.literal('else')) + stmt).optional()).map(If)
        args = expr // comma
        resolve = sb.alternate((tok(sb.literal('.')) + id_), expr.between(lbrack, rbrack), args.optional().between(lpar, rpar))
        lvalue = (sb.alternate(litval, id_.map(Variable), expr.between(lpar, rpar).select(1)) + resolve.some(True)).map(
            lambda result:
                ( Call(result)
                    if result[-1][-1][0] == '('
                    else Accessor(result) )
                if len(result) == 2 else result[0]
        )
        factor = (unaryop.some(True) + lvalue).map(
            lambda result:
                UnaryOp(result) if len(result) == 2 else result[0]
        )
        labo = lambda r: r[0] if len(r) == 1 else LeftAssocBinOp(r)
        term = (factor + (mulop + factor).some(True)).map(labo)
        sum_ = (term + (addop + term).some(True)).map(labo)
        shift = (sum_ + (shiftop + sum_).some(True)).map(labo)
        bits = (shift + (bitop + shift).some(True)).map(labo)
        comp = (bits + (compop + bits).some(True)).map(labo)
        rvalue = (comp + (logicop + comp).some(True)).map(labo)
        expr <<= ((lvalue + assign).some(True) + rvalue).map(
            lambda result:
                Assignment(result) if len(result) == 2 else result[0]
        ) | rvalue
        """Need the above alternation because there can be a conflict between
        assign and compop where "value ==" will match one (lvalue + assign)
        and then fail the following rvalue.
        """
        type_ = (id_ + (lbrack + rbrack).some()).map(Typespec)
        var = (id_ + (assign + expr).optional())
        vardec = sb.concatenate(type_, (var // comma), semicolon).map(Vardec)
        block = stmt.some(True).between(tok(sb.literal('{')), tok(sb.literal('}')), True).map(BlockStmt)
        stmt << sb.alternate(
            if_, while_, for_, break_, continue_, return_, vardec, block,
            (expr + semicolon).map(ExprStmt))
        params = (type_ + id_) // comma # [[type, id], [type, id], ...]
        funcdef = sb.concatenate(type_, id_, lpar.suppress(), params.optional().wrap(), rpar.suppress(), block).map(Funcdef)
        program = (ignore.suppress() + (funcdef | vardec).some()).extract()
    return program

def test():
    p = gen_parser()
    txt = '''
void main(){
    output = 0.5;
    for (i1 i = 0; i < 10; i += 1;) {
        if (i == 2) {
            output[2] = 10;
        } else if (i == 1) {
            output[5] = 10;
        } else {
            output[i] = i;
        }
    }
    i = 0;
}
'''
    m = p(txt)
    print(m)
    print(f'{m.position} vs {len(txt)}')
    print(txt[m.position:])
    ctx = compiler.Context()
    print('Program:\n\n', '\n'.join(ctx.eval_program(m.result)))
    
    vm = pyvm.VM()
    vm.pre_init = ctx.pre_init
    vm.functions.update({f'{key[0]}({",".join(key[1])})': value.code for key, value in ctx.funcs.items() if value.code is not None})
    vm.global_vars = [0] * len(ctx.vars)
    vm.run()
    print(vm.global_vars[ctx.vars['output'].location[1]])
    assert not vm.stack

if __name__ == '__main__':
    test()
