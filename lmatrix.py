"""
labeled matrix 
"""
import numpy as np
from numpy import ndarray

np.set_printoptions(precision=3, suppress=True)

class LMatrix(ndarray):
    """Labeled Matrix class"""

    def __new__(cls, labels= [], data = None, labels_synonyms = []):
        """labels: list of str
        matrix: None by default,
        if presented, be `np.array` like object
        """
        #if matrix is presented, pass it to the new function
        if data is not None:
            rn,cn = data.shape
            
            #the row count and col count should equal
            if rn != cn:
                raise ValueError("not square matrix")
            elif rn != len(labels):
                raise ValueError("label size and matrix dimension not match")
            obj = np.asarray(data).view(cls)
        else:
            #else, only init the labels
            obj = ndarray.__new__(cls, (len(labels), len(labels)))

        obj.labels = labels

        for syn in labels_synonyms:
            setattr(obj, syn, obj.labels)

        #label to index mapping
        obj.label2index_mapping = dict((l,i) for i,l in enumerate(obj.labels))
        
        #label to index mapping
        obj.index2label_mapping = dict((i,l) for i,l in enumerate(obj.labels))

        return obj
    
    def get_label(self, idx):
        return self.index2label_mapping.get(idx, "unkown")

        
    def __array_finalize__(self, obj):    
        if obj is None: return
        
        self.labels = getattr(obj, "labels", None)
        self.label2index_mapping = getattr(obj, "label2index_mapping", None)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            s1,s2 = key
            if isinstance(s1,str) and isinstance(s2,str):#s1 and s2 are both labels
                i1 = self.label2index_mapping[s1]
                i2 = self.label2index_mapping[s2]
                return self.view(ndarray)[i1,i2]
            elif isinstance(s1,str):#s1 is string
                i1 = self.label2index_mapping[s1]
                return self.view(ndarray)[i1,s2]
            elif isinstance(s2,str):#s2 is string
                i2 = self.label2index_mapping[s2]
                return self.view(ndarray)[s1,i2]

                
        return (self.view(ndarray)[key]).view(self.__class__)
    
    def __setitem__(self, key, item):
        if isinstance(key, tuple):
            s1, s2 = key
            if isinstance(s1,str) and isinstance(s2,str):#s1 and s2 are both labels
                i1 = self.label2index_mapping[s1]
                i2 = self.label2index_mapping[s2]
                super(LMatrix, self).__setitem__((i1, i2), item)
                return
            elif isinstance(s1,str):#s1 is label
                i1 = self.label2index_mapping[s1]
                super(LMatrix, self).__setitem__((i1, s2), item)
                return
            elif isinstance(s2,str):#s2 is label
                i2 = self.label2index_mapping[s2]
                super(LMatrix, self).__setitem__((s1, i2), item)
                return
        #otherwise
        super(LMatrix, self).__setitem__(key, item)

def main():
    labels = ["l1", "l2", "l3"]
    m = LMatrix(labels, labels_synonyms = ["states"])

    m[0,:] = [1,2,3]
    m["l2",:] = [4,5,6]
    
    m["l3","l1"] = 7
    m[:,"l3"] = [3,6,9]
    m["l3",:2] = [7,8]

    print m["l1",:]
    print m[:,'l2']
    print m
    print m.labels
    print m.states
if __name__ == "__main__":
    main()
