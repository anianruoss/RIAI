def relu(x):
    return [max(0, x[0]), max(0, x[1])]


def mult(s, x):
    return [min(s * x[0], s * x[1]), max(s * x[0], s * x[1])]


def add(x, y):
    return [x[0] + y[0], x[1] + y[1]]


def addS(s, x):
    return [s + x[0], s + x[1]]


def net(x, y):
    n1 = relu(add(mult(-1, x), addS(2, y)))
    n2 = relu(add(x, mult(-2, y)))

    return relu(add(n1, n2))


def meet(x, y):
    return [min(x[0], y[0]), max(x[1], y[1])]


print('Run interval analysis with [0, 2] x [0, 1]')
print(net([0, 2], [0, 1]))
print('Run interval analysis with ([0, 1] U [1, 2]) x [0, 1]')
print(meet(net([0, 1], [0, 1]), net([1, 2], [0, 1])))
