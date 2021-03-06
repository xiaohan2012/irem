from common import *
from lmatrix import LMatrix
from fb import backward_prob_table

class BackwardAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.pi = hashdict([("s", 0.85), ("t",0.16)]) 
        
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

    def test_result_one(self):
        """test whether the backward prob table is calculated correctly, for the first case"""
        expected = LMatrix(rlabels = ["s", "t"],
                           clabels = range(4),
                           data = np.array([
                               [0.13315, 0.25610, 0.47000, 1],
                               [0.12729, 0.24870, 0.49000, 1]
                           ])
        )

        actual, obs_prob = backward_prob_table(("A", "B", "B", "A"), self.A, self.B, self.pi)
        
        self.assertEqual(expected, actual)
        self.assertAlmostEqual(obs_prob, 0.055451099999999996)
        
    def test_result_two(self):
        """test whether the backward prob table is calculated correctly, for the second case"""
        expected = LMatrix(rlabels = ["s", "t"],
                           clabels = range(3),
                           data = np.array([
                               [0.24210, 0.53000, 1],
                               [0.25070, 0.51000, 1]
                           ])
        )

        actual, obs_prob = backward_prob_table(("B", "A", "B"), self.A, self.B, self.pi)
        self.assertAlmostEqual(obs_prob, 0.14352700000000002)

if __name__ == '__main__':
    unittest.main()        