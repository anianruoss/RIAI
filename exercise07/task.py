from collections import defaultdict
from random import uniform


class ZonotopeContext:
    """Zonotope context.

    A zonotope context creates and transforms affine forms.  Each form is
    represented as a dictionary.  The items of the dictionary are pairs::

        (i, a)

    where ``i`` is the index of a noise symbol and ``a`` is the respective
    coefficient for that noise symbol.  The free coefficient is represented
    as the noise symbol with index 0, that is the pair ``(0, a)``.

    Parameters
    ----------
    n : int
        Number of noise symbols in pre-existing affineforms.
    """

    def __init__(self, n=0):
        self.n = n

    def make(self, m, d):
        """Create an affine form.

        The form is specified with two parameters.

        Parameters
        ----------
        m : float
            Midpoint.
        d : float
            Maximum distance from midpoint.

        Returns
        -------
        affine form

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

    def extrema(self, f):
        """Compute extrema of an affine form.

        Parameters
        ----------
        f : affine form

        Returns
        -------
        float tuple
            The minimum and the maximum.

        Examples
        --------
        >>> ctx = ZonotopeContext(3)
        >>> ctx.extrema({0: 14.0, 1: 0.25, 2: 0.5, 3: 0.75})
        (12.5, 15.5)
        """
        a = sum(abs(f[i]) for i in f if i != 0)

        return f[0] - a, f[0] + a

    def dot(self, C, F):
        """Compute dot product.

        Parameters
        ----------
        C : float list
        F : affine form list

        Returns
        -------
        affine form

        Examples
        --------
        >>> ctx = ZonotopeContext()
        >>> C = [1.0,
        ...      2.0,
        ...      3.0]
        >>> F = [ctx.make(1.0, 0.25),
        ...      ctx.make(2.0, 0.25),
        ...      ctx.make(3.0, 0.25)]
        >>> sorted(ctx.dot(C, F).items())
        [(0, 14.0), (1, 0.25), (2, 0.5), (3, 0.75)]
        """
        assert 0 < len(C)
        assert len(C) == len(F)
        t = defaultdict(float)

        for c, f in zip(C, F):
            for i in f:
                t[i] += c * f[i]

        return t

    def relu(self, f):
        """Compute ReLU.

        Parameters
        ----------
        f : affine form

        Returns
        -------
        affine form

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
        l, u = self.extrema(f)

        if u <= 0:
            return self.make(0, 0)
        if l >= 0:
            return f

        lambda_ = u / (u - l)
        mu_ = - lambda_ * l / 2

        return self.dot([lambda_, mu_], [f, self.make(1., 1.)])


def network(ctx, x0, x1, x2):
    y1 = ctx.relu(ctx.dot([2., -1., 1.], [x0, x1, x2]))
    y2 = ctx.relu(ctx.dot([0., 1., -2.], [x0, x1, x2]))
    return ctx.relu(ctx.dot([1., 1.], [y1, y2]))


def problem4():
    ctx = ZonotopeContext()
    x0 = ctx.make(1., 0.)
    x1 = ctx.make(1., 1.)
    x2 = ctx.make(0.5, 0.5)

    return ctx, network(ctx, x0, x1, x2)


def test_problem4(extrema, n=1000):
    for i in range(n):
        ctx = ZonotopeContext()
        x0 = ctx.make(1., 0.)
        x1 = ctx.make(uniform(0., 2.), 0.)
        x2 = ctx.make(uniform(0., 1.), 0.)
        output = network(ctx, x0, x1, x2)
        lo, hi = ctx.extrema(output)

        assert extrema[0] <= lo and hi <= extrema[1]


ctx, f = problem4()
test_problem4(ctx.extrema(f))
print('form: {}'.format(sorted(f.items())))
print(' min: {}, max: {}'.format(*ctx.extrema(f)))
