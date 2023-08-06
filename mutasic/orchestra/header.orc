/*
Builtin functions that can be implemented in-language.
*/

fm sin(fm phase) {
    fm s = phase + 0; /* Copy value */
    for (i1 j = 0; j < block_size; j += 1;) {
        s[j] = sin(s[j]);
    }
    return s;
}

fm asin(fm leg) {
    fm theta = leg + 0;
    for (i1 j = 0; j < block_size; j += 1;) {
        theta[j] = asin(theta[j]);
    }
    return theta;
}

f1 cos(f1 phase) {
    phase = pi / 2 - phase;
    return sin(phase);
}

fm cos(fm phase) {
    fm c = phase + 0;
    for (i1 j = 0; j < block_size; j += 1;) {
        c[j] = cos(c[j]);
    }
    return c;
}

f1 acos(f1 leg) {
    f1 as = asin(leg);
    return pi / 2 - as;
}

fm acos(fm phase) {
    fm c = phase + 0;
    for (i1 j = 0; j < block_size; j += 1;) {
        c[j] = acos(c[j]);
    }
    return c;
}

fm atan2(fm y, fm x) {
    x = x + 0;
    for (i1 j = 0; j < block_size; j += 1;) {
        x[j] = atan2(y[j], x[j]);
    }
    return x;
}

f1 abs(f1 arg) {
    if (arg < 0) {
        return -arg;
    }
    return arg;
}

f1 abs(c1 arg) {
    f1 real = arg.real;
    f1 imag = arg.imag;
    return sqrt(real * real + imag * imag);
}

f1 arg(c1 num) {
    return atan2(num.imag, num.real);
}


