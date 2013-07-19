from fb import forward_prob_table

from lmatrix import LMatrix

from common import *

class ForwardAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.pi = (("s", 0.85), ("t", 0.16)) #two values don't sum to 1, this is because we want to accomondate to the rounding error in Moss's lecture
        
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
        """test whether the forward prob table is calculated correctly, for the first case"""
        expected = LMatrix(rlabels = ["s", "t"],
                           clabels = range(4),
                           data = np.array([
                               [0.34, 0.066, 0.02118, 0.00625],
                               [0.075, 0.155, 0.09285, 0.04919]
                           ])
        )

        actual = forward_prob_table(("A", "B", "B", "A"), self.A, self.B, self.pi)
        
        self.assertEqual(expected, actual)

    def test_result_two(self):
        """test whether the forward prob table is calculated correctly, for the second case"""
        expected = LMatrix(rlabels = ["s", "t"],
                           clabels = range(3),
                           data = np.array([
                               [0.51, 0.0644, 0.0209],
                               [0.08, 0.2145, 0.1190]
                           ])
        )

        actual = forward_prob_table(("B", "A", "B"), self.A, self.B, self.pi)
        
        self.assertEqual(expected, actual)
        
class YetAnotherTest(unittest.TestCase):
    pass
    
if __name__ == '__main__':
    unittest.main()