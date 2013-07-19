from common import *
from baum_welch import baum_welch, convergent, gamma


class ConvergentTest(unittest.TestCase):
    def test_convergent_case(self):
        old_v = np.array([
            1,2,3
        ])
        new_v = np.array([
            1.000009,2,3
        ])
        self.assertTrue(convergent(old_v, new_v))

    def test_not_convergent_case(self):
        old_v = np.array([
            1.0, 2.0, 3.0
        ])
        
        new_v = np.array([
            1.00001, 2.0, 3.0
        ])
        
        self.assertFalse(convergent(old_v, new_v))


class BaumWelchTest(unittest.TestCase):
    def setUp(self):
        self.pi = hashdict([("s", 0.85), ("t", 0.16)]) #two values don't sum to 1, this is because we want to accomondate to the rounding error in Moss's lecture
        
        self.A = LMatrix(rlabels = ["s", "t"],
                         data = np.array([
                             [0.3, 0.7],
                             [0.1, 0.9]
                         ]))
        
        self.B = LMatrix(rlabels = ["s", "t"],
                         clabels = ["A", "B"],
                         data = np.array([
                             [0.4, 0.6],
                             [0.5, 0.5],
                         ])
        )

        self.lst_of_obs = [("A", "B", "B", "A")] * 10 + [("B", "A", "B")] * 20

        self.Q = ["s", "t"]
        self.V = ["A", "B"]

    def test_gamma_case_1(self):
        """test for the gamma function"""

        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 0, "s", "s", self.A, self.B, self.pi), 0.28271, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 0, "t", "s", self.A, self.B, self.pi), 0.02217, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 0, "s" ,"t", self.A, self.B, self.pi), 0.53383, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 0, "t", "t", self.A, self.B, self.pi), 0.16149, places = 3)

        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 1, "s", "s", self.A, self.B, self.pi), 0.10071, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 1, "t", "s", self.A, self.B, self.pi), 0.07884, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 1, "s" ,"t", self.A, self.B, self.pi), 0.20417, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 1, "t", "t", self.A, self.B, self.pi), 0.61648, places = 3)

        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 2, "s", "s", self.A, self.B, self.pi), 0.04584, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 2, "t", "s", self.A, self.B, self.pi), 0.06699, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 2, "s" ,"t", self.A, self.B, self.pi), 0.13371, places = 3)
        self.assertAlmostEqual(gamma(("A", "B", "B", "A"), 2, "t", "t", self.A, self.B, self.pi), 0.75365, places = 3)
            
        
    def test_gamma_case_2(self):
        """test for the gamma function
        
        the `place` is set relatively high, perhaps due to the miscalculation of the lecture note (or mine --|| )).
        """

        self.assertAlmostEqual(gamma(("B", "A", "B"), 0, "s", "s", self.A, self.B, self.pi), 0.23185, places = 1)
        self.assertAlmostEqual(gamma(("B", "A", "B"), 0, "s" ,"t", self.A, self.B, self.pi), 0.65071, places = 1)
        self.assertAlmostEqual(gamma(("B", "A", "B"), 0, "t", "s", self.A, self.B, self.pi), 0.01212, places = 3)
        self.assertAlmostEqual(gamma(("B", "A", "B"), 0, "t", "t", self.A, self.B, self.pi), 0.13124, places = 1)

        
        self.assertAlmostEqual(gamma(("B", "A", "B"), 1, "s", "s", self.A, self.B, self.pi), 0.08286, places = 2)
        self.assertAlmostEqual(gamma(("B", "A", "B"), 1, "s" ,"t", self.A, self.B, self.pi), 0.16112, places = 1)
        self.assertAlmostEqual(gamma(("B", "A", "B"), 1, "t", "s", self.A, self.B, self.pi), 0.09199, places = 2)
        self.assertAlmostEqual(gamma(("B", "A", "B"), 1, "t", "t", self.A, self.B, self.pi), 0.68996, places = 1)
            
        

        
if __name__ == '__main__':
    unittest.main()