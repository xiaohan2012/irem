from common import *
from fb import forward_backward, convergent


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
        

class ForwardBackwardAlgorithmTest(unittest.TestCase):
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
                         ]))

        self.lst_of_obs = [
            ("3", "1", "3"),
            ("1", "3", "1"),
            ("3", "3", "1")
        ]

        self.Q = {"H", "C"}
        self.V = {"3", "1"}


    def test_if_runnable(self):
        A, B = forward_backward(self.lst_of_obs, self.V, self.Q, self.A, self.B)
        print A,B

        
if __name__ == '__main__':
    unittest.main()