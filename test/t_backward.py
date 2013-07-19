from common import *
from lmatrix import LMatrix
from fb import backward_prob_table, START, END

class BackwardAlgorithmTest(unittest.TestCase):
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
                           clabels = range(3),
                           data = np.array([
                               [0.0315, 0.045, 0.6],
                               [0.029, 0.245, 0.1],
                               [0.0315, 0.29, 0.7]
                           ])
        )
        
        actual = backward_prob_table(("K3", "K2", "K1"), self.A, self.B, self.pi)
        
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()        