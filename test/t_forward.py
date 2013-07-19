from fb import forward_prob_table, START, END

from lmatrix import LMatrix

from common import *

class ForwardAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.pi = (("S1", 1), ("S2", 0.0))
        
        self.A = LMatrix(rlabels = ["S1", "S2"],
                         data = np.array([
                             [0.7, 0.3],
                             [0.5, 0.5]
                         ]))
        
        self.B = LMatrix(rlabels = ["S1", "S2"],
                         clabels = ["K1", "K2", "K3"],
                         data = np.array([
                             [0.6, 0.1, 0.3],
                             [0.1, 0.7, 0.2],
                         ])
        )

    def test_result(self):
        """test whether the forward prob table is calculated correctly"""
        expected = LMatrix(rlabels = ["S1", "S2", "Total"],
                           clabels = range(4),
                           data = np.array([
                               [1, 0.21, 0.0462, 0.021294],
                               [0.0, 0.09, 0.0378, 0.010206],
                               [1, 0.3, 0.084, 0.0315]
                           ])
        )

        actual = forward_prob_table(("K3", "K2", "K1"), self.A, self.B, self.pi)
        
        self.assertEqual(expected, actual)

class YetAnotherTest(unittest.TestCase):
    pass
    
if __name__ == '__main__':
    unittest.main()