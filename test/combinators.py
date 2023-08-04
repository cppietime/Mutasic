from crackrat.combinators.core import *

def expect_result(m, result):
    assert not m.is_error and m.result == result, m

def expect_error(m):
    assert m.is_error, m

def test_singles():
    """digits, alpha, literal, regex, success, failure"""
    d = digits()
    m = d('1234')
    expect_result(m, '1234')
    m = d('123abc')
    expect_result(m, '123')
    m = d('abcd')
    expect_error(m)
    a = alpha()
    m = a('hello')
    expect_result(m, 'hello')
    m = a('1234')
    expect_error(m)
    m = a('abc123')
    expect_result(m, 'abc')
    l = literal('token')
    m = l('token')
    expect_result(m, 'token')
    m = l('nah')
    expect_error(m)
    m = l('tokenkek')
    expect_result(m, 'token')
    reg = regex('(qed)+')
    m = reg('qedqedqeqe')
    expect_result(m, 'qedqed')
    s = success(60)
    m = s('anything')
    expect_result(m, 60)
    f = failure(1, 2)
    m = f('anything')
    assert m.is_error and m.error_reason == 1 and m.error_message == 2, m

def test_top_level_combinators():
    """allternate, concatenate, longest"""
    a = literal('a')
    b = literal('b')
    c = literal('c')
    alt = alternate(a, b, c)
    for tok in ('a', 'b', 'c'):
        m = alt(tok)
        expect_result(m, tok)
    m = alt('kek')
    expect_error(m)
    seq = concatenate(a, b, c)
    m = seq('abc')
    expect_result(m, ['a', 'b', 'c'])
    m = seq('ababc')
    expect_error(m)
    left = regex('abc')
    right = regex('abcabc')
    first = alternate(left, right)
    f_match = first('abcabcabc')
    expect_result(f_match, 'abc')
    longer = longest(left, right)
    l_match = longer('abcabcabc')
    expect_result(l_match, 'abcabc')

def test_binaries():
    """+, -, |, &, ^, @, /, //, between"""
    abcs = regex('abc+')
    defs = regex('def+')
    seq = abcs + defs
    m = seq('abcccdef')
    expect_result(m, ['abccc', 'def'])
    m = seq('abcc')
    expect_error(m)
    m = seq('abccabcc')
    expect_error(m)
    either = abcs | defs
    m = either('abcccdef')
    expect_result(m, 'abccc')
    m = either('def')
    expect_result(m, 'def')
    endc = regex('.*ccc')
    both = abcs & endc
    m = both('abccc')
    expect_result(m, 'abccc')
    m = both('abcc')
    expect_error(m)
    m = both('qedcccc')
    expect_error(m)
    xor = abcs ^ endc
    m = xor('abcccdeccc')
    expect_error(m)
    m = xor('abccdcc')
    expect_result(m, 'abcc')
    sub = abcs - endc
    m = sub('abcccdeccc')
    expect_error(m)
    m = sub('abccdcc')
    expect_result(m, 'abcc')
    m = sub('adccc')
    expect_error(m)
    longer = abcs @ endc
    m = longer('abcccdeccc')
    expect_result(m, 'abcccdeccc')
    m = longer('abccde')
    expect_result(m, 'abcc')
    m = longer('abdccc')
    expect_result(m, 'abdccc')
    sep = alpha() / literal(',')
    m = sep('a,b,c,d')
    expect_result(m, ['a', 'b', 'c', 'd'])
    m = sep('')
    expect_result(m, [])
    m = sep('a')
    expect_result(m, ['a'])
    m = sep('a,b,')
    assert not m.is_error and m.result == ['a', 'b'] and m.position == 4, m
    sep = alpha() // literal(',')
    m = sep('a,b,')
    assert not m.is_error and m.result == ['a', 'b'] and m.position == 3, m
    bet = abcs.between(literal('['), literal(']'))
    m = bet('[abccc]')
    expect_result(m, ['[', 'abccc', ']'])
    m = bet('[abcdef]')
    expect_error(m)
    bets = abcs.between(literal('['), literal(']'), suppress=True)
    m = bets('[abccc]')
    expect_result(m, ['abccc'])
    some = abcs.some()
    m = some('abcccccabc')
    expect_result(m, ['abccccc', 'abc'])
    m = some('qeqe')
    expect_result(m, [])
    somes = abcs.some(True)
    m = somes('kek')
    expect_result(m, None)

def test_lazy():
    l = ['abc']
    def func():
        return literal(l[0])
    p = lazy(func)
    m = p('abc')
    expect_result(m, 'abc')
    m = p('def')
    expect_error(m)
    l[0] = 'def'
    m = p('def')
    expect_result(m, 'def')
    m = p('abc')
    expect_error(m)

def test_memo():
    memo = {}
    lst = (regex('[^,\\[\\]]*') // literal(',')).between(literal('['), literal(']')).memoize(memo)
    assert len(memo) == 0
    m = lst('[a,b,c]')
    expect_result(m, ['[', ['a', 'b', 'c'], ']'])
    assert len(memo) == 1
    m = lst('[a,b,c]')
    expect_result(m, ['[', ['a', 'b', 'c'], ']'])
    assert len(memo) == 1
    memo[('test', 0)] = ParserState('test', 100, result='force')
    m = lst('test')
    expect_result(m, 'force')

def test_recursion():
    """Test the following (non-left) recursive grammar:
    LIST = lbrack, [ ELEMS ], rbrack;
    ELEMS = ELEM, { comma, ELEM };
    ELEM = word | LIST;
    """
    elements = forward()
    lst = elements.optional().between(literal('['), literal(']')).flatten()
    element = alpha() | lst
    elements << (element / literal(','))
    m = lst('[abc]')
    expect_result(m, ['[', 'abc', ']'])
    m = lst('[]')
    expect_result(m, ['[', ']'])
    m = lst('[abc,def,abc]')
    expect_result(m, ['[', 'abc', 'def', 'abc', ']'])
    m = lst('[left,[inner,inner],middle,[],right]')
    expect_result(m, ['[', 'left', ['[', 'inner', 'inner', ']'], 'middle', ['[', ']'], 'right', ']'])

def test_map():
    integer = digits().map(int)
    factor = forward()
    def reduce_product(arr):
        p = 1
        for n in arr:
            p *= n
        return p
    term = (factor.wrap() + (literal('*').suppress() + factor).some(True).flatten()).flatten().map(reduce_product)
    expr = (term.wrap() + (literal('+').suppress() + term).some(True).flatten()).flatten().map(sum)
    factor << (integer | (expr.between(literal('('), literal(')'), True)).select(0))
    """print(term('1*2*3'))
    print(expr('1*2+1'))
    print(expr('(1*2+1)'))
    print(expr('1+1*2+1*(1+1)'))"""
    math = '(1*2+1)*(1+2*(1+3))+7*6+(0)'
    m = expr(math)
    # print(m)
    expect_result(m, eval(math))

def test_tokenizing():
    with Memoizer():
        space = ws()
        def tok(parser):
            return (parser + space.optional()).select(0)
        ident = tok(identifier())
        plus = tok(literal('+'))
        sc = tok(literal(';'))
        stmt = ((ident // plus) + sc).select(0)
    m = stmt('abc + _alpha + n2 ;')
    expect_result(m, ['abc', '_alpha', 'n2'])
    m = stmt('no+space;')
    expect_result(m, ['no', 'space'])
    m = stmt('_l_o_t_s_     +\nof\t\t+ space \n ;')
    expect_result(m, ['_l_o_t_s_', 'of', 'space'])
    prog = (space.some().suppress() + stmt.many()).flatten()
    m = prog('''
a + bcd + ef_69;
my_ + man_
;
oneline
+
anotherline; and + new
+ thing;''')
    expect_result(m, [['a', 'bcd', 'ef_69'], ['my_', 'man_'], ['oneline', 'anotherline'], ['and', 'new', 'thing']])

def test_big():
    """Test the grammar:
    PROGRAM = { FUNCDEF | VARDEC };
    FUNCDEF = TYPE, id, lparen, [ PARAMS ], rparen, BLOCK;
    TYPE = id, [ lbrack, rbrack ];
    PARAMS = TYPE, id, { comma, TYPE, id };
    BLOCK = lbrace, { STMT }, rbrace;
    STMT = EXPR, semicolon | IF | WHILE | FOR | BREAK | CONTINUE | RETURN | BLOCK | VARDEC;
    EXPR = { LVALUE, assign }, RVALUE;
    LVALUE = VALUE, { RESOLVE }; (* need to add more to this but idk exactly what yet *)
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
    expr = forward()
    stmt = forward()
    with Memoizer():
        comment = regex('\\/\\*([^*]|\n|\*[^/])*\\*\\/')
        ignore = alternate(ws(), comment).some()
        def tok(parser):
            return (parser + ignore).select(0)
        semicolon = tok(literal(';'))
        lpar = tok(literal('('))
        rpar = tok(literal(')'))
        comma = tok(literal(','))
        id_ = tok(identifier())
        lbrack = tok(literal('['))
        rbrack = tok(literal(']'))
        num = tok(number())
        litval = tok(number())
        unaryop = tok(alternate(*map(literal, ['-', '!', '~'])))
        addop = tok(alternate(*map(literal, ['+', '-'])))
        mulop = tok(alternate(*map(literal, ['*', '/', '%'])))
        shiftop = tok(alternate(*map(literal, ['<<', '>>'])))
        bitop = tok(alternate(*map(literal, ['|', '&', '^'])))
        compop = tok(alternate(*map(literal, ['==', '!=', '<=', '>=', '<', '>'])))
        logicop = tok(alternate(*map(literal, ['||', '&&'])))
        assign = tok(alternate(*map(literal, ['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='])))
        return_ = concatenate(tok(literal('return')).suppress(), expr.optional(), semicolon.suppress())
        break_ = (tok(literal('break')) + semicolon.suppress()).select(0)
        continue_ = (tok(literal('continue')) + semicolon.suppress()).select(0)
        for_ = concatenate(tok(literal('for')), lpar.suppress(), expr, rpar.suppress(), stmt)
        while_ = concatenate(tok(literal('while')), lpar.suppress(), expr, rpar.suppress(), stmt)
        if_ = concatenate(tok(literal('if')), lpar.suppress(), expr, rpar.suppress(), stmt, (tok(literal('else')) + stmt).optional())
        args = expr // comma
        resolve = alternate((tok(literal('.')) + id_), expr.between(lbrack, rbrack), args.optional().between(lpar, rpar))
        lvalue = (alternate(litval, id_, expr.between(lpar, rpar).select(1)) + resolve.some(True)).extract()
        factor = (unaryop.some(True) + lvalue).extract() # factor = [-, ..., lvalue]
        term = (factor // mulop).extract() # term = [factor, ..., factor] ... lvalue
        sum_ = (term // addop).extract() # sum = [term, ..., term] ... lvalue
        shift = (sum_ // shiftop).extract()
        bits = (shift // bitop).extract()
        comp = (bits // compop).extract()
        rvalue = (comp // logicop).extract()
        expr << ((lvalue + assign.suppress()).some(True) + rvalue).extract() # [[lvalue, lvalue...], rvalue] or rvalue
        type_ = (id_ + (lbrack + rbrack).optional()) # [id] or [id, [, ]]
        var = (id_ + (assign.suppress() + expr).some(True).extract()).flatten()
        vardec = concatenate(type_, (var // comma).extract(), semicolon.suppress()).flatten()
        block = stmt.some(True).between(tok(literal('{')), tok(literal('}')), True)
        stmt << alternate(expr + semicolon, if_, while_, for_, block, break_, continue_, return_, vardec)
        params = (type_.extract() + id_) // comma
        funcdef = concatenate(type_.extract(), id_, lpar.suppress(), params.optional(), rpar.suppress(), block.extract())
        program = ignore.suppress() + (funcdef | vardec).some()
    test_input = '''
/* I am a comment */
type variable;
type variable;
/* And I spanmany lines*/
type[] var = v, x = 1 + 2 * (2+3|4);
return func(t1 arg1, t2 arg2){
    let x = 7, y, z = 24().x[27];
    if (true) 72; else if (false) 48; else 69;
    x.y = /* c*/ abc();
    x.y.z[a] = kek.u(27)
    ;
    {
        let u = 70;
    }
}
'''
    m = program(test_input)
    print(m)
    print(f'{m.position} vs {len(test_input)}')
    print(litval('7'))

def main():
    test_singles()
    test_top_level_combinators()
    test_binaries()
    test_lazy()
    test_memo()
    test_recursion()
    test_map()
    test_tokenizing()
    test_big()
    print('No errors')

if __name__ == '__main__':
    main()
