from common import *
from train import init_estimate
from util import read_annotation

class InitEstimateTest(unittest.TestCase):
    def setUp(self):
        self.obs_list = read_annotation(open("data/annotated.json", "r"))

    def test_runnable(self):
        A, B, pi = init_estimate(self.obs_list)
        print A, B, pi
        
if __name__ == '__main__':
    unittest.main()
        