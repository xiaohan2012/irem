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
        _, actual, _ = viterbi(self.obs, self.A, self.B, self.pi)
        expected = LMatrix(("H", "L"),
                           xrange(len(self.obs)),
                           data = np.array([
                               [ -2.737, -5.474, -8.211, -11.533, -14.007, -17.329, -19.54, -22.862, -25.657],
                               [ -3.322, -6.059, -8.796, -10.948, -14.007, -16.481, -19.54, -22.014, -24.487]
                           ])
        )
        for s in actual.rlabels:
            for t in actual.clabels:
                self.assertAlmostEqual(actual[s,t], expected[s,t], 3)

    def test_state_sequence(self):
        actual, _, _ = viterbi(self.obs, self.A, self.B, self.pi)
        expected = ("H", "H", "H", "L", "L", "L", "L", "L", "L")
        self.assertEqual(actual, expected)

    def test_backtrace(self):
        _, _, actual = viterbi(self.obs, self.A, self.B, self.pi)
        expected = [{'H': 'H', 'L': 'H'}, {'H': 'H', 'L': 'H'}, {'H': 'H', 'L': 'H'}, {'H': 'L', 'L': 'L'}, {'H': 'H', 'L': 'L'}, {'H': 'L', 'L': 'L'}, {'H': 'H', 'L': 'L'}, {'H': 'L', 'L': 'L'}]
        self.assertEqual(actual, expected)
        
if __name__ == '__main__':
    unittest.main()