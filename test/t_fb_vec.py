"""
test for the vectorized version of forward-backward prob calculation
"""

from common import *
from fb_vec import forward_vec

class ForwardTest(unittest.TestCase):
        
    def test_fb_table(self):
        obs, A, B, pi = weather_data()
        
        expected = LMatrix(rlabels = ["rain", "sunny"],
                           clabels = range(len(obs)),
                           data = np.array([
                               [0.8182, 0.8834, 0.1907, 0.7308, 0.8673],
                               [0.1818, 0.1166, 0.8093, 0.2692,0.1327]
                           ])
        )

        actual, _ = forward_vec(obs, A, B, pi)
        for s in A.rlabels:
            for i in xrange(len(obs)):
                self.assertAlmostEqual(expected[s,i], actual[s,i], 4)
        
if __name__ == '__main__':
    unittest.main()