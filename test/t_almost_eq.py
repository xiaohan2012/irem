from common import *
from util import almost_eq

class AlmostEqualTest(unittest.TestCase):
    def test_equal_case(self):
        """identical case"""
        a = np.array([
            [1.0,2.333 ],
        ])
        b = np.array([
            [1.0,2.333],
        ])

        self.assertTrue(almost_eq(a, b))

    def test_almost_equal_case(self):
        """almost equal, treated as equal"""
        a = np.array([
            [1.0,2.3331 ],
        ])
        b = np.array([
            [1.0,2.333],
        ])

        self.assertFalse(almost_eq(a, b))

    def test_not_equal_case(self):
        """almost equal, but not treated as equal"""
        a = np.array([
            [1.0,2.333001 ],
        ])
        b = np.array([
            [1.0,2.333],
        ])

        self.assertTrue(almost_eq(a, b))
        

if __name__ == '__main__':
    unittest.main()
        