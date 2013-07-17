from common import *
from lmatrix import LMatrix
from fb import backward_prob_table, START, END

class BackwardAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.A = LMatrix(rlabels = [START, "H", "C", END],
                         data = np.array([
                             [0, 0.8, 0.2, 0],
                             [0, 0.7, 0.3, 0.5],
                             [0, 0.4, 0.6, 0.5],
                             [0, 0.0, 0.0, 0]
                         ]))
        
        self.B = LMatrix(rlabels = [START, "H", "C", END],
                         clabels = ["1", "3"],
                         data = np.array([
                             [0, 0],
                             [0.2, 0.4],
                             [0.5, 0.1],
                             [0, 0]
                         ])
        )

    def test_result(self):
        """test whether the forward prob table is calculated correctly"""
        expected_table = LMatrix(rlabels = [START, "H", "C", END],
                           clabels = range(3),
                           data = np.array([
                               [0, 0, 0],
                               [0.0382, 0.155, 0.5],
                               [0.0452, 0.11, 0.5],
                               [0, 0, 0]
                           ])
        )
        expected_prob = 0.013132
        
        actual_prob, actual_table = backward_prob_table(("3", "1", "3"), self.A, self.B)
        
        self.assertAlmostEqual(expected_prob, actual_prob)
        self.assertEqual(expected_table, actual_table)

if __name__ == '__main__':
    unittest.main()        