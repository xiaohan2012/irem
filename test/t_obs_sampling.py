from common import *
from util import sample_observations_from_file

class ObservationSamplingEqualTest(unittest.TestCase):
    def test_sample_count(self):
        """sample 2000, test if the result is of length 2000"""
        count = 2000
        samples = sample_observations_from_file(open("data/ingredients5000.json", "r"), count)
        self.assertEqual(len(samples), count)

if __name__ == '__main__':
    unittest.main()
