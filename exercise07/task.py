from collections import defaultdict
from random import uniform


class ZonotopeContext:
    """
    A zonotope context creates and transforms affine forms. Each form is
    represented as a dictionary. The items of the dictionary are pairs: (i, a)
    where `i` is the index of a noise symbol and `a` is the respective
    coefficient for that noise symbol. The free coefficient is represented
    as the noise symbol with index 0, that is the pair `(0, a)`.
    """

    def __init__(self, n=0):
        """
        :type n: int
        :param n: number of noise symbols in pre-existing affine forms
        """
        self.n = n

    def make(self, m, d):
        """
        Creates and affine form
        :type m: float
        :param m: midpoint
        :type d: float
        :param d: maximum distance from midpoint
        :rtype: defaultdict
        :return: affine form

        Examples
        --------
        >>> ctx = ZonotopeContext()
        >>> sorted(ctx.make(1.0, 0.25).items())
        [(0, 1.0), (1, 0.25)]
        """
        assert d >= 0.
        z = defaultdict(float, {0: float(m)})

        if d > 0.:
            self.n += 1
            z[self.n] = float(d)

        return z

    @staticmethod
    def extrema(affine_form):
        """
        Compute extrema of affine form
        :type affine_form: defaultdict
        :param affine_form: affine form
        :rtype: float tuple
        :return: minimum and maximum

        Examples
        --------
        >>> ctx = ZonotopeContext(3)
        >>> ctx.extrema({0: 14.0, 1: 0.25, 2: 0.5, 3: 0.75})
        (12.5, 15.5)
        """
        a = sum(abs(affine_form[i]) for i in affine_form if i != 0)

        return affine_form[0] - a, affine_form[0] + a

    @staticmethod
    def dot(coefficients, affine_forms):
        """
        Compute dot product
        :type coefficients: list of float
        :param coefficients:
        :type affine_forms: list of defaultdict
        :param affine_forms:
        :rtype: defaultdict
        :return: new affine form

        Examples
        --------
        >>> ctx = ZonotopeContext()
        >>> C = [1.0, 2.0, 3.0]
        >>> F = [ctx.make(1.0, 0.25), ctx.make(2.0, 0.25), ctx.make(3.0, 0.25)]
        >>> sorted(ctx.dot(C, F).items())
        [(0, 14.0), (1, 0.25), (2, 0.5), (3, 0.75)]
        """
        assert 0 < len(coefficients)
        assert len(coefficients) == len(affine_forms)
        t = defaultdict(float)

        for coefficient, affine_form in zip(coefficients, affine_forms):
            for i in affine_form:
                t[i] += coefficient * affine_form[i]

        return t

    def relu(self, affine_form):
        """
        compute ReLU
        :type affine_form: defaultdict
        :param affine_form: affine form
        :rtype: defaultdict
        :return: new affine form

        Examples
        --------
        >>> ctx = ZonotopeContext(3)
        >>> sorted(ctx.relu({0: +14.0, 1: 0.25, 2: 0.5, 3: 0.75}).items())
        [(0, 14.0), (1, 0.25), (2, 0.5), (3, 0.75)]
        >>> sorted(ctx.relu({0: -14.0, 1: 0.25, 2: 0.5, 3: 0.75}).items())
        [(0, 0.0)]
        >>> sorted(ctx.relu({0: 0.000, 1: 0.25, 2: 0.5, 3: 0.75}).items())
        [(0, 0.375), (1, 0.125), (2, 0.25), (3, 0.375), (4, 0.375)]
        """
        lower, upper = self.extrema(affine_form)

        if upper <= 0:
            return self.make(0, 0)
        if lower >= 0:
            return affine_form

        lambda_ = upper / (upper - lower)
        mu_ = - lambda_ * lower / 2

        return self.dot([lambda_, mu_], [affine_form, self.make(1., 1.)])


def network(context, x0, x1, x2):
    y1 = context.relu(context.dot([2., -1., 1.], [x0, x1, x2]))
    y2 = context.relu(context.dot([0., 1., -2.], [x0, x1, x2]))

    return context.relu(context.dot([1., 1.], [y1, y2]))


def problem4():
    context = ZonotopeContext()
    x0 = context.make(1., 0.)
    x1 = context.make(1., 1.)
    x2 = context.make(0.5, 0.5)

    return context, network(context, x0, x1, x2)


def test_problem4(extrema, n=1000):
    for i in range(n):
        context = ZonotopeContext()
        x0 = context.make(1., 0.)
        x1 = context.make(uniform(0., 2.), 0.)
        x2 = context.make(uniform(0., 1.), 0.)
        output = network(context, x0, x1, x2)
        lo, hi = context.extrema(output)

        assert extrema[0] <= lo and hi <= extrema[1]


ctx, f = problem4()
test_problem4(ctx.extrema(f))
print('form: {}'.format(sorted(f.items())))
print('min: {}, max: {}'.format(*ctx.extrema(f)))
