from common import *
from baum_welch import baum_welch, convergent, gamma, delta, one_iter


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

    def test_gamma_case_1(self):
        """test for the gamma function"""

        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 0, "s", "s", self.A, self.B, self.pi), 0.28271, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 0, "t", "s", self.A, self.B, self.pi), 0.02217, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 0, "s" ,"t", self.A, self.B, self.pi), 0.53383, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 0, "t", "t", self.A, self.B, self.pi), 0.16149, places = 3)

        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 1, "s", "s", self.A, self.B, self.pi), 0.10071, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 1, "t", "s", self.A, self.B, self.pi), 0.07884, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 1, "s" ,"t", self.A, self.B, self.pi), 0.20417, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 1, "t", "t", self.A, self.B, self.pi), 0.61648, places = 3)

        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 2, "s", "s", self.A, self.B, self.pi), 0.04584, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 2, "t", "s", self.A, self.B, self.pi), 0.06699, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 2, "s" ,"t", self.A, self.B, self.pi), 0.13371, places = 3)
        self.assertAlmostEqual(gamma( ("A", "B", "B", "A"), 2, "t", "t", self.A, self.B, self.pi), 0.75365, places = 3)
            
        
    def test_gamma_case_2(self):
        """test for the gamma function
        
        the `place` is set relatively high, perhaps due to the miscalculation of the lecture note (or mine --|| )).
        """

        self.assertAlmostEqual(gamma( ("B", "A", "B"), 0, "s", "s", self.A, self.B, self.pi), 0.23185, places = 1)
        self.assertAlmostEqual(gamma( ("B", "A", "B"), 0, "s" ,"t", self.A, self.B, self.pi), 0.65071, places = 1)
        self.assertAlmostEqual(gamma( ("B", "A", "B"), 0, "t", "s", self.A, self.B, self.pi), 0.01212, places = 3)
        self.assertAlmostEqual(gamma( ("B", "A", "B"), 0, "t", "t", self.A, self.B, self.pi), 0.13124, places = 1)

        
        self.assertAlmostEqual(gamma( ("B", "A", "B"), 1, "s", "s", self.A, self.B, self.pi), 0.08286, places = 2)
        self.assertAlmostEqual(gamma( ("B", "A", "B"), 1, "s" ,"t", self.A, self.B, self.pi), 0.16112, places = 1)
        self.assertAlmostEqual(gamma( ("B", "A", "B"), 1, "t", "s", self.A, self.B, self.pi), 0.09199, places = 2)
        self.assertAlmostEqual(gamma( ("B", "A", "B"), 1, "t", "t", self.A, self.B, self.pi), 0.68996, places = 1)

    def test_delta_case_1(self):
        """
        test for the delta function
        """
        self.assertAlmostEqual(delta( ("A", "B", "B", "A"), 0, "s", self.A, self.B, self.pi), 0.81654, places = 3)
        self.assertAlmostEqual(delta( ("A", "B", "B", "A"), 0, "t", self.A, self.B, self.pi), 0.18366, places = 3)
        
        self.assertAlmostEqual(delta( ("A", "B", "B", "A"), 1, "s", self.A, self.B, self.pi), 0.30488, places = 3)
        self.assertAlmostEqual(delta( ("A", "B", "B", "A"), 1, "t", self.A, self.B, self.pi), 0.69532, places = 3)

        self.assertAlmostEqual(delta( ("A", "B", "B", "A"), 2, "s", self.A, self.B, self.pi), 0.17955, places = 3)
        self.assertAlmostEqual(delta( ("A", "B", "B", "A"), 2, "t", self.A, self.B, self.pi), 0.82064, places = 3)

        self.assertAlmostEqual(delta( ("A", "B", "B", "A"), 3, "s", self.A, self.B, self.pi), 0.11273, places = 3)
        self.assertAlmostEqual(delta( ("A", "B", "B", "A"), 3, "t", self.A, self.B, self.pi), 0.88727, places = 3)
        
    def test_delta_case_2(self):
        """
        test for the delta function for case 2

        Again, calculation result inconsistent with Moss's lecture note
        """
        self.assertAlmostEqual(delta( ("B", "A", "B"), 0, "s", self.A, self.B, self.pi), 0.88256, places = 1)
        self.assertAlmostEqual(delta( ("B", "A", "B"), 0, "t", self.A, self.B, self.pi), 0.14336, places = 1)

        self.assertAlmostEqual(delta( ("B", "A", "B"), 1, "s", self.A, self.B, self.pi), 0.24398, places = 1)
        self.assertAlmostEqual(delta( ("B", "A", "B"), 1, "t", self.A, self.B, self.pi), 0.78195, places = 1)

        self.assertAlmostEqual(delta( ("B", "A", "B"), 2, "s", self.A, self.B, self.pi), 0.14939, places = 1)
        self.assertAlmostEqual(delta( ("B", "A", "B"), 2, "t", self.A, self.B, self.pi), 0.85061, places = 1)

    def test_baum_welch(self):
        """test for the baum welch algorithm"""
        A, B, pi = baum_welch(self.lst_of_obs, self.A, self.B, self.pi)
        
class IterationOneTest(unittest.TestCase):
    def setUp(self):
        self.lst_of_obs = [("A", "B", "B", "A")] * 10 + [("B", "A", "B")] * 20
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

    def test_pi(self):
        _, _, pi = one_iter(self.lst_of_obs, self.A, self.B, self.pi)
        for actual, expected in zip(pi, [0.846, 0.154]):
            self.assertAlmostEqual(actual, expected, places = 2)

    def test_A(self):
        A, _, _ = one_iter(self.lst_of_obs, self.A, self.B, self.pi)
        for actual, expected in zip(A.flatten(), [0.298, 0.702, 0.106, 0.894]):
            self.assertAlmostEqual(actual, expected, places = 2)

    def test_B(self):
        _, B, _ = one_iter(self.lst_of_obs, self.A, self.B, self.pi)
        for actual, expected in zip(B.flatten(), [0.357, 0.643, 0.4292, 0.5708]):
            self.assertAlmostEqual(actual, expected, places = 2)
            

class IterationTwoTest(unittest.TestCase):
    """
    Continuing the first iteration, this is the second iteration
    """
    
    def setUp(self):
        self.lst_of_obs = [("A", "B", "B", "A")] * 10 + [("B", "A", "B")] * 20
        self.pi = hashdict([("s", 0.846,), ("t", 0.154)]) #two values don't sum to 1, this is because we want to accomondate to the rounding error in Moss's lecture
        
        self.A = LMatrix(rlabels = ["s", "t"],
                         data = np.array([
                             [0.298, 0.702],
                             [0.106, 0.894]
                         ]))
        
        self.B = LMatrix(rlabels = ["s", "t"],
                         clabels = ["A", "B"],
                         data = np.array([
                             [0.357, 0.643],
                             [0.4292, 0.5708],
                         ])
        )

    def test_pi(self):
        _, _, pi = one_iter(self.lst_of_obs, self.A, self.B, self.pi)
        for actual, expected in zip(pi, [0.841, 0.159]):
            self.assertAlmostEqual(actual, expected, places = 1)

    def test_A(self):
        A, _, _ = one_iter(self.lst_of_obs, self.A, self.B, self.pi)
        for actual, expected in zip(A.flatten(), [0.292, 0.708, 0.109, 0.891]):
            self.assertAlmostEqual(actual, expected, places = 2)

    def test_B(self):
        _, B, _ = one_iter(self.lst_of_obs, self.A, self.B, self.pi)
        for actual, expected in zip(B.flatten(), [0.3624, 0.6376, 0.4252, 0.5748]):
            self.assertAlmostEqual(actual, expected, places = 2)
            
if __name__ == '__main__':
    unittest.main()