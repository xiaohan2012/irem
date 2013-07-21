from common import *
from train import init_estimate
from util import read_annotation, get_V

class InitEstimateTest(unittest.TestCase):
    def setUp(self):
        self.annotation_list = read_annotation(open("data/annotated.json", "r"))

    def test_prob_sum_to_one(self):
        """
        wether the prob dist matches the sum-to-one property
        """
        A, B, pi = init_estimate(("O", "B", "C"), self.annotation_list)
        
        for i in A.sum(1):
            self.assertAlmostEqual(i, 1)
            
        for i in B.sum(1):
            self.assertAlmostEqual(i, 1)
            
        self.assertAlmostEqual(sum(pi.values()), 1)
            
class InitEstimateWithExtraVocabTest(unittest.TestCase):
    """the same with the above, except incorporating extra output vocabulary"""
    
if __name__ == '__main__':
    unittest.main()
        