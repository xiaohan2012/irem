from common import *
from lmatrix import LMatrix

class LMatrixTest(unittest.TestCase):
    def test_no_data_given(self):
        m = LMatrix(rlabels = ["1", "2", "3"])
        for i in m.flatten():
            self.assertEqual(i, 0)
        
class IdenticalLabelsTest(unittest.TestCase):
    """test for identical labels for row and column"""
    
    def setUp(self):
        labels = ["l1", "l2", "l3"]
        self.m = LMatrix(labels, data=np.array([[1,2,3],[4,5,6],[7,8,9]]))
        
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
    
    def test_eq(self):
        labels = ["l1", "l2", "l3"]
        other = LMatrix(labels, data=np.array([[1,2,3],[4,5,6],[7,8,9]]))
        self.assertEqual(other, self.m)

    def test_ne(self):
        labels = ["l1", "l2", "l3"]
        other = LMatrix(labels, data=np.array([[1,2,3],[4,5,6],[7,8,10]]))
        self.assertNotEqual(other, self.m)
        
class UnidenticalLabelsTest(unittest.TestCase):
    """test for identical labels for row and column"""
    
    def setUp(self):
        rlabels = ["r1", "r2"]
        clabels = ["c1", "c2", "c3"]
        self.m = LMatrix(rlabels, clabels, data = np.array([[1,2,3],[4,5,6]]))

    def test_numeric_index_setitem(self):
        """set item using numeric index"""
        self.m[0,:] = [1,2,3]
        actual = np.all(self.m[0,:] == np.array([1,2,3]))
        expected = 1
        self.assertEqual(actual, expected)

    def test_label_index_setitem(self):
        """set item using label index"""
        self.m["r1",:] = [4,5,6]
        actual = np.all(self.m["r1",:] == np.array([4, 5, 6]))
        expected = 1
        self.assertEqual(actual, expected)

    def test_label_index_getitem(self):
        """set item using label index"""
        actual = self.m["r2", "c2"]
        expected = 5
        self.assertEqual(actual, expected)

class MalformMatrixTest(unittest.TestCase):
    def test_dimension_unmatch(self):
        rlabels = ["r1", "r2"]
        clabels = ["c1", "c2", "c3"]
        self.assertRaises(ValueError, LMatrix, rlabels, clabels, data = np.array([[1,2,3]]))

if __name__ == '__main__':
    unittest.main()
        

