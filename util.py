import numpy as np

def almost_eq(this, that, decimal = 5):
    return np.all((this - that) < 10 ** (-decimal))

import collections
import functools

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

class hashdict(dict):
    """
    hashable dict implementation, suitable for use as a key into
    other dicts.

        >>> h1 = hashdict({"apples": 1, "bananas":2})
        >>> h2 = hashdict({"bananas": 3, "mangoes": 5})
        >>> h1+h2
        hashdict(apples=1, bananas=3, mangoes=5)
        >>> d1 = {}
        >>> d1[h1] = "salad"
        >>> d1[h1]
        'salad'
        >>> d1[h2]
        Traceback (most recent call last):
        ...
        KeyError: hashdict(bananas=3, mangoes=5)

    based on answers from
       http://stackoverflow.com/questions/1151658/python-hashable-dicts

    """
    def __key(self):
        return tuple(sorted(self.items()))
    def __repr__(self):
        return "{0}({1})".format(self.__class__.__name__,
            ", ".join("{0}={1}".format(
                    str(i[0]),repr(i[1])) for i in self.__key()))

    def __hash__(self):
        return hash(self.__key())
    def __setitem__(self, key, value):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def __delitem__(self, key):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def clear(self):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def pop(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def popitem(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def setdefault(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def update(self, *args, **kwargs):
        raise TypeError("{0} does not support item assignment"
                         .format(self.__class__.__name__))
    def __add__(self, right):
        result = hashdict(self)
        dict.update(result, right)
        return result

from collections import namedtuple
from simplejson import load, loads

AnnotatedWord = namedtuple("AnnotatedWord", "word tag")

def read_annotation(f):
    """
    (file like object) -> list of ((text, tag), (text, tag), (text, tag), ....)
    """
    
    return [tuple([(AnnotatedWord(a["text"], a["tags"][0][0].upper() if len(a["tags"]) == 1 and a["tags"][0] in ("begin", "continue") else "O"))
             for a in sent])
            for sent in loads(f.read())]
    
def sample_observations_from_file(f, n=2000):
    """sample `n` observations from recipe file, `f`"""
    import random
    return map(lambda r: tuple(r), random.sample(load(f), n))

def get_V(annotation_list, obs_list = []):
    """given the annotation list and the observation list, return the output vocabulary"""
    return  set( (pair.word for sent in annotation_list for pair in sent) ).union(set( (ob for obs in obs_list for ob in obs) ) )

def load_HMM(iter_id = "28"):
    from cPickle import load
    
    A = load(open("param_snapshot/%s_A.mat" % iter_id, "r"))
    B = load(open("param_snapshot/%s_B.mat" % iter_id, "r"))
    pi = load(open("param_snapshot/%s_pi.mat" % iter_id, "r"))

    return A, B, pi
    
def test():
    import doctest
    doctest.testmod()
    
def main():
    ans = read_annotation(open("data/annotated.json", "r"))
    for sent in ans:
        for a in sent:
            print a.word,"/",a.tag,",",
        print
        
if __name__ == '__main__':
    main()