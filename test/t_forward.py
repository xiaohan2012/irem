from fb import *

from lmatrix import LMatrix

from common import *

class ForwardAlgorithmTest(unittest.TestCase):
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
        expected = LMatrix(rlabels = [START, "H", "C", END],
                           clabels = range(3),
                           data = np.array([
                               [0, 0, 0],
                               [0.32, 0.0464, 0.021632],
                               [0.02, 0.05400, 0.004632],
                               [0, 0, 0.013132]
                           ])
        )

        actual = forward_prob_table(obs = ["3", "1", "3"], A = self.A, B = self.B)
        
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()