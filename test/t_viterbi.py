from common import *
from viterbi import viterbi

class DNASequenceTest(unittest.TestCase):
    def setUp(self):
        self.A = LMatrix(
            ("H", "L"),
            data = np.array([
                [0.5, 0.5],
                [0.4, 0.6]
            ])
        )        
        self.B = LMatrix(
            ("H", "L"),
            ("A", "C", "G", "T"),
            data = np.array([
                [0.2, 0.3, 0.3, 0.2],
                [0.3, 0.2, 0.2, 0.3]
            ])
        )

        self.pi = hashdict([("H", 0.5), ("L", 0.5)])

        self.obs = tuple(list("GGCACTGAA"))

    def test_trellis(self):
        _, trellis, _ = viterbi(self.obs, self.A, self.B, self.pi)
        print trellis

if __name__ == '__main__':
    unittest.main()