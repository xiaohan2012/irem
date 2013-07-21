from common import *
from train import discounted

from collections import Counter

class DiscountTest(unittest.TestCase):
    def setUp(self):
        self.B = LMatrix(("s1", "s2"),
                         ("often", "frequent", "infrequent", "unusual"),
                         data = np.array([
                             [30., 15., 3., 2.],
                             [14., 5., 0., 1.]
                         ])
        )
        self.V = set( ("often", "frequent", "infrequent", "unusual", "qweqwe", "sdvqwe"))
        self.freq = Counter(["often"] * 50 + ["frequent"] * 25 + ["infrequent"] * 3 + ["unusual"] * 4 + ["qweqwe", "sdvqwe", "sdvqwe"])

    def test_runnable(self):
        actual = discounted(self.B, self.V, self.freq)
        expected = LMatrix( ("s1", "s2"),
                 ("often", "frequent", "infrequent", "unusual", "qweqwe", "sdvqwe"),
                 data = np.array([
                     [0.59, 0.29, 0.05, 0.03, 0.013333333333333334, 0.02666666666666667],
                     [0.675, 0.225, 0.037499999999999978, 0.025, 0.012499999999999992, 0.024999999999999984]
                 ])
        )
        
        for s in actual.rlabels:
            for o in actual.clabels:
                self.assertAlmostEqual(actual[s,o], expected[s,o])


if __name__ == '__main__':
    unittest.main()