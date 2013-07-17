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
        

if __name__ == '__main__':
    unittest.main()