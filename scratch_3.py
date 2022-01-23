def least_difference(a, b, c):
    diff1 = a + b
    diff2 = b + c
    diff3 = c + a
    return max(diff1, diff2, diff3)


least_difference(1, 2, 3)
