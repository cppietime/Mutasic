from crackrat.combinators.core import *

def test_singles():
    """digits, alpha, literal, regex, success, failure"""
    d = digits()
    d_match = d('1234')
    assert not d_match.is_error and d_match.result == '1234'
    d_match = d('123abc')
    assert not d_match.is_error and d_match.result == '123'
    d_match = d('abcd')
    assert d_match.is_error
    a = alpha()
    a_match = a('hello')
    assert not a_match.is_error and a_match.result == 'hello'
    a_match = a('1234')
    assert a_match.is_error
    a_match = a('abc123')
    assert not a_match.is_error and a_match.result == 'abc'
    l = literal('token')
    l_match = l('token')
    assert not l_match.is_error and l_match.result == 'token'
    l_match = l('nah')
    assert l_match.is_error
    l_match = l('tokenkek')
    assert not l_match.is_error and l_match.result == 'token'
    reg = regex('(qed)+')
    r_match = reg('qedqedqeqe')
    assert not r_match.is_error and r_match.result == 'qedqed'
    s = success(60)
    s_match = s('anything')
    assert not s_match.is_error and s_match.result == 60
    f = failure(1, 2)
    f_match = f('anything')
    assert f_match.is_error and f_match.error_reason == 1 and f_match.error_message == 2

def test_top_level_combinators():
    """allternate, concatenate, longest"""
    a = literal('a')
    b = literal('b')
    c = literal('c')
    alt = alternate(a, b, c)
    for tok in ('a', 'b', 'c'):
        m = alt(tok)
        assert not m.is_error and m.result == tok
    m = alt('kek')
    assert m.is_error
    seq = concatenate(a, b, c)
    m = seq('abc')
    assert not m.is_error and m.result == ['a', 'b', 'c']
    m = seq('ababc')
    assert m.is_error
    left = regex('abc')
    right = regex('abcabc')
    first = alternate(left, right)
    f_match = first('abcabcabc')
    assert not f_match.is_error and f_match.result == 'abc'
    longer = longest(left, right)
    l_match = longer('abcabcabc')
    assert not l_match.is_error and l_match.result == 'abcabc'

def test_binaries():
    """+, -, |, &, ^, @, /, //, between"""
    abcs = regex('abc+')
    defs = regex('def+')
    seq = abcs + defs
    m = seq('abcccdef')
    assert not m.is_error and m.result == ['abccc', 'def']
    either = abcs | defs
    m = either('abcccdef')
    assert not m.is_error and m.result == 'abccc'
    m = either('def')
    assert not m.is_error and m.result == 'def'
    endc = regex('.*ccc')
    both = abcs & endc
    m = both('abccc')
    assert not m.is_error and m.result == 'abccc'
    m = both('abcc')
    assert m.is_error
    m = both('qedcccc')
    assert m.is_error
    xor = abcs ^ endc
    m = xor('abcccdeccc')
    assert m.is_error
    m = xor('abccdcc')
    assert not m.is_error and m.result == 'abcc'
    sub = abcs - endc
    m = sub('abcccdeccc')
    assert m.is_error
    m = sub('abccdcc')
    assert not m.is_error and m.result == 'abcc'
    m = sub('adccc')
    assert m.is_error
    longer = abcs @ endc
    m = longer('abcccdeccc')
    assert not m.is_error and m.result == 'abcccdeccc'
    m = longer('abccde')
    assert not m.is_error and m.result == 'abcc'
    m = longer('abdccc')
    assert not m.is_error and m.result == 'abdccc'
    sep = alpha() / literal(',')
    m = sep('a,b,c,d')
    assert not m.is_error and m.result == ['a', 'b', 'c', 'd']
    m = sep('')
    assert not m.is_error and m.result == []
    m = sep('a')
    assert not m.is_error and m.result == ['a']
    m = sep('a,b,')
    assert not m.is_error and m.result == ['a', 'b'] and m.position == 4
    sep = alpha() // literal(',')
    m = sep('a,b,')
    assert not m.is_error and m.result == ['a', 'b'] and m.position == 3
    bet = abcs.between(literal('['), literal(']'))
    m = bet('[abccc]')
    assert not m.is_error and m.result == ['[', 'abccc', ']']
    m = bet('[abcdef]')
    assert m.is_error

# TODO transformations, recursion, lazy

def main():
    test_singles()
    test_top_level_combinators()
    test_binaries()
    print('No errors')

if __name__ == '__main__':
    main()
