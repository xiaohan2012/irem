from common import *
from train import init_estimate, word_freq
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
    """
    the same with the above, except incorporating extra output vocabulary

    it may take a while
    """
    
    def setUp(self):
        self.annotation_list = read_annotation(open("data/annotated.json", "r"))
        
        #get the observation list
        self.obs_list = sample_observations_from_file(open("data/ingredients5000.json", "r"), 2500)
        
        self.freq = word_freq(self.annotation_list, self.obs_list)
        
        #get Q and V
        self.Q = ("O", "B", "C")
        self.V = get_V(self.annotation_list, self.obs_list)


    def test_prob_sum_to_one(self):
        """
        wether the prob dist matches the sum-to-one property
        """
        A, B, pi = init_estimate(self.Q, self.annotation_list,  discount = True, V = self.V, vfreq = self.freq)
        
        for i in A.sum(1):
            self.assertAlmostEqual(i, 1)
            
        for i in B.sum(1):
            self.assertAlmostEqual(i, 1)
            
        self.assertAlmostEqual(sum(pi.values()), 1)

    def test_smoothed(self):
        """all vals in B should be greater than 0, aka, smoothed value"""
        _, B, _ = init_estimate(self.Q, self.annotation_list,  discount = True, V = self.V, vfreq = self.freq)
        self.assertTrue( (B > 0).all() )
        
if __name__ == '__main__':
    unittest.main()
        