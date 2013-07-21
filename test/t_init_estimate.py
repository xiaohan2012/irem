from common import *
from train import init_estimate
from util import read_annotation

class InitEstimateTest(unittest.TestCase):
    def setUp(self):
        self.obs_list = read_annotation(open("data/annotated.json", "r"))

    def test_prob_sum_to_one(self):
        """
        wether the prob dist matches the sum-to-one property
        """
        
        A, B, pi = init_estimate(("O", "B", "C"), self.obs_list)
        
        for i in A.sum(1):
            self.assertAlmostEqual(i, 1)
            
        for i in B.sum(1):
            self.assertAlmostEqual(i, 1)
            
        self.assertAlmostEqual(sum(pi.values()), 1)
            
        
if __name__ == '__main__':
    unittest.main()
        