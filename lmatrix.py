"""
labeled matrix 
"""
import numpy as np
from numpy import ndarray

np.set_printoptions(precision=3, suppress=True)

class LMatrix(ndarray):
    """Labeled Matrix class"""

    def __new__(cls, rlabels= [], clabels = None, data = None):
        """
        rlabels: list of hashable obj, like str, for the rows,
        
        clabels: list of hashable obj, like str, for the cols.
        if clabels not presented, it is the same as rlabels
        
        matrix: None by default,
        if presented, be `np.array` like object
        
        labels_synonyms: iother names for attr labels
        """
        if clabels is None:
            clabels = rlabels
            
        #if matrix is presented, pass it to the new function
        if data is not None:
            r_cnt,c_cnt = data.shape
            
            #the row count and col count should equal
            if r_cnt != len(rlabels) and c_cnt != len(clabels):
                raise ValueError("label size and matrix dimension not match ( %dx%d required, %dx%d given)" %(len(rlabels),
                                                                                                             len(clabels),
                                                                                                             r_cnt,
                                                                                                             c_cnt))
            obj = np.asarray(data).view(cls)
        else:
            obj = ndarray.__new__(cls, (len(rlabels), len(clabels)))

        obj.rlabels = rlabels
        obj.clabels = clabels

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
            if not isinstance(s1,int) and not isinstance(s2,int):#s1 and s2 neither int
                i1 = self.label2index_mapping[s1]
                i2 = self.label2index_mapping[s2]
                return self.view(ndarray)[i1,i2]
            elif not isinstance(s1,int):#s1 is not int
                i1 = self.label2index_mapping[s1]
                return self.view(ndarray)[i1,s2]
            elif not isinstance(s2,int):#s2 is not int
                i2 = self.label2index_mapping[s2]
                return self.view(ndarray)[s1,i2]

                
        return (self.view(ndarray)[key]).view(self.__class__)
    
    def __setitem__(self, key, item):
        if isinstance(key, tuple):
            s1, s2 = key
            if not isinstance(s1,int) and not isinstance(s2,int):#s1 and s2 neither int
                i1 = self.label2index_mapping[s1]
                i2 = self.label2index_mapping[s2]
                super(LMatrix, self).__setitem__((i1, i2), item)
                return
            elif not isinstance(s1,int):#s1 is not int
                i1 = self.label2index_mapping[s1]
                super(LMatrix, self).__setitem__((i1, s2), item)
                return
            elif not isinstance(s2,int):#s2 is not int
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
