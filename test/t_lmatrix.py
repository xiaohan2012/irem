from common import *
from lmatrix import LMatrix

class LMatrixTest(unittest.TestCase):
    def setUp(self):
        labels = ["l1", "l2", "l3"]
        self.m = LMatrix(labels, labels_synonyms = ["states"])

    def test_numeric_index_setitem(self):
        """set item using numeric index"""
        self.m[0,:] = [1,2,3]
        actual = np.all(self.m[0,:] == np.array([1,2,3]))
        expected = 1
        self.assertEqual(actual, expected)

    def test_label_index_setitem(self):
        """set item using label index"""
        self.m["l2",:] = [4,5,6]
        actual = np.all(self.m["l2",:] == np.array([4, 5, 6]))
        expected = 1
        self.assertEqual(actual, expected)
        
    def test_label_synonym(self):
        """get item using numeric index"""
        self.assertEqual(self.m.states, self.m.labels)


if __name__ == '__main__':
    unittest.main()
        

