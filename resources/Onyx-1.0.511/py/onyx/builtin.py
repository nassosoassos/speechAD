###########################################################################
#
# File:         builtin.py
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Some useful specializations of builtin objects
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 The Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
#
###########################################################################

"""
    Low-level containers and mixin classes, candidates for being builtin.
"""

from itertools import izip as _izip

# private helpers

def _wrapped_repr(cls, self):
    return cls.__name__ + '(' + super(cls, self).__repr__() + ')'

def _operand_error_msg(op, obj1, obj2):
    return "unsupported operand type(s) for %s: %s and %s" % (op, repr(type(obj1).__name__), repr(type(obj2).__name__),)

# used to override mutating methods
def _mutability_error(self, *dummy_args):
    raise TypeError("%s object does not support being modified" % (repr(type(self).__name__),))

def _attribute_error(self, *dummy_args):
    raise AttributeError("%s object does not support modification of its attributes" % (repr(type(self).__name__),))
    
class _has_introspector(object):
    """
    Introspecting helper for has_descriptor descriptor used by the
    mix-in class 'has'.
    """
    __slots__ = ('other',)
    def __init__(self, other):
        self.other = other
    def __getattr__(self, attr):
        return hasattr(self.other, attr)

class _has_descriptor(object):
    """
    Descriptor class for the .hasattr non-data descriptor used by
    the mix-in class 'has'.
    """
    __slots__ = tuple()
    def __get__(self, obj, typ=None):
        return _has_introspector(obj)


class has(object):
    """
    Mix-in baseclass which provides the property .hasattr (a
    descriptor) that provides attribute syntax to introspectively to
    see if an instance has a particular attribute.

    >>> x = type('X', (has,), {})()
    >>> x.hasattr.foo
    False
    >>> x.foo = 23
    >>> x.hasattr.foo
    True
    >>> del x.foo
    >>> x.hasattr.foo
    False
    """
    __slots__ = tuple()
    hasattr = _has_descriptor()


# This little class covers up a problem between the docstring for the built-in
# dict method 'update' and Sphinx.  The update docstring has an unescaped ** in
# it, which Sphinx warns about as an "Inline strong start-string without
# end-string".  In this class, we override the update method and give the
# overriding method an escaped version of the string.  We then use this class
# everywhere we want to derived from dict.
class _rst_clean_dict(dict):
    def update(self, E, **kwargs):
        super(_rst_clean_dict, self).update(E, **kwargs)
    update.__doc__ = dict.update.__doc__.replace('**', r"\*\*")


class attrdict(_rst_clean_dict, has):
    """
    Creates a specialization of dict with the feature that dotted
    attribute syntax can be used to access items whose keys are
    strings that are valid Python identifiers.  Also, the attribute
    .hasattr supports attribute syntax to see if the instance has a
    particular attribute.  

    If a subclass has a __slots__ class attribute with symbol names
    that begin with underscore, these symbols will be treated as
    regular instance attributes and will not appear in the dictionary.

    >>> d = attrdict({'foo':3})
    >>> d.foo
    3

    >>> d.bar = 4
    >>> d['bar']
    4

    >>> del d.foo
    >>> d
    attrdict({'bar': 4})

    >>> d.hasattr.bar
    True
    >>> d.hasattr.foo
    False

    >>> foo = type('foo', (attrdict,), {'__slots__':('_foobar',)})
    >>> f = foo()
    >>> f._foobar = 1
    >>> f
    attrdict({})
    >>> f._foobar
    1
    >>> del f._foobar
    >>> f._foobar
    Traceback (most recent call last):
      ...
    AttributeError: _foobar
    """

    __slots__ = tuple()
    def __setattr__(self, name, value):
        assert isinstance(name, str) and len(name) >= 1
        if name.startswith('_'):
            super(attrdict, self).__setattr__(name, value)
        else:
            self[name] = value
    def __getattr__(self, name):
        assert isinstance(name, str) and len(name) >= 1
        if name.startswith('_'):
            # unlike __getattr__, using __getattribute__ emits a useful error message on failure
            return super(attrdict, self).__getattribute__(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(self._attr_error_msg(name))
        
    def __delattr__(self, name):
        assert isinstance(name, str) and len(name) >= 1
        if name.startswith('_'):
            super(attrdict, self).__delattr__(name)
        else:
            try:
                del self[name]
            except KeyError:
                raise AttributeError(self._attr_error_msg(name))
    def __repr__(self):
        return _wrapped_repr(attrdict, self)
    def copy(self):
        return type(self)(self.iteritems())
    # note: we get fromkeys() for free because it's a class method of dict!
    def _attr_error_msg(self, name):
        return "%s object has no attribute (or no key) %s" % (repr(type(self).__name__), repr(name),)

class frozentuple(tuple): 
    """
    Creates a specialization of tuple with the feature that it ensures
    that all its contained items are immutable.

    A frozentuple is created and used just like a tuple, and it can
    always be used as a dictionary key or as a set member.  Unlike
    tuple the constructor will raise a TypeError if any of the
    (recursively) contained items are mutable, i.e. cannot be hashed.

    >>> t = tuple('abc')
    >>> f = frozentuple(t)
    >>> f + frozentuple('def')
    frozentuple(('a', 'b', 'c', 'd', 'e', 'f'))

    >>> tuple(f)
    ('a', 'b', 'c')

    >>> list(f)
    ['a', 'b', 'c']

    >>> 3 * f
    frozentuple(('a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'))

    Show that copy construction is cheap

    >>> frozentuple(f) is f
    True

    Empty frozen tuple is ok

    >>> frozentuple()
    frozentuple(())

    Each item in the constructor iterable must be immutable

    >>> frozentuple((('a', 'b', 'c'), [1, 2])) #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
        frozentuple((('a', 'b', 'c'), [1, 2]))
      File "<stdin>", line ###, in __init__
    TypeError: ...unhashable...
    """
    __slots__ = tuple()
    def __new__(cls, arg=()):
        return arg if type(arg) is frozentuple else super(frozentuple, cls).__new__(cls, arg)
    def __init__(self, *args):
        hash(self)
    def __add__(self, other):
        if not isinstance(other, frozentuple):
            raise TypeError(_operand_error_msg('+', self, other))
        return frozentuple(super(frozentuple, self).__add__(other))
    def __radd__(self, other):
        assert not isinstance(other, frozentuple)
        raise TypeError(_operand_error_msg('+', other, self))
    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError(_operand_error_msg('*', self, other))
        return frozentuple(super(frozentuple, self).__mul__(other))
    def __rmul__(self, other):
        if not isinstance(other, int):
            raise TypeError(_operand_error_msg('*', other, self))
        return frozentuple(super(frozentuple, self).__rmul__(other))

    def __repr__(self):
        return _wrapped_repr(frozentuple, self)


class frozenlist(list):
    """
    Creates a specialization of list with the features that it cannot be
    modified and it ensures that all its contained items are immutable.

    A frozenlist is created and used like a list.  It can always be used as a
    dictionary key or as a set member.  Unlike a list, attempts to modify the
    object will raise a TypeError or AttributeError, and the constructor will
    raise a TypeError if any of the (recursively) contained items are mutable,
    i.e. cannot be hashed.

    >>> t = list('abc')
    >>> f = frozenlist(t)
    >>> f + frozenlist('def')
    frozenlist(['a', 'b', 'c', 'd', 'e', 'f'])

    >>> list(f)
    ['a', 'b', 'c']

    >>> tuple(f)
    ('a', 'b', 'c')
    
    >>> 3 * f
    frozenlist(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])

    >>> frozenlist(f) is f
    True

    Empty frozen list is ok

    >>> frozenlist()
    frozenlist([])

    Can't put a list into a set

    >>> s = set()
    >>> s.add(t)  #doctest: +ELLIPSIS
    Traceback (most recent call last):
       ...
    TypeError: ...unhashable...

    Can put a frozenlist into a set

    >>> s.add(frozenlist(f))
    >>> s
    set([frozenlist(['a', 'b', 'c'])])

    Each item in the constructor iterable must be immutable

    >>> frozenlist(['abc', [1, 2]])  #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
        frozenlist(['abc', [1, 2]])
      File "<stdin>", line ###, in __init__
      File "<stdin>", line ###, in <genexpr>
    TypeError: ...unhashable...
    """

    __slots__ = ('_hash',)
    def __new__(cls, arg=()):
        return arg if type(arg) is frozenlist else super(frozenlist, cls).__new__(cls, arg)
    def __init__(self, *arg):
        if not hasattr(self, '_hash'):
            super(frozenlist, self).__init__(*arg)
            hashaccum = reduce(int.__xor__, (hash(item) for item in self), 0x00)
            object.__setattr__(self, '_hash', hashaccum)        
    def __hash__(self):
        return self._hash

    def __add__(self, other):
        if not isinstance(other, frozenlist):
            raise TypeError(_operand_error_msg('+', self, other))
        return frozenlist(super(frozenlist, self).__add__(other))
    def __radd__(self, other):
        assert not isinstance(other, frozenlist)
        raise TypeError(_operand_error_msg('+', other, self))
    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError(_operand_error_msg('*', self, other))
        return frozenlist(super(frozenlist, self).__mul__(other))
    def __rmul__(self, other):
        if not isinstance(other, int):
            raise TypeError(_operand_error_msg('*', other, self))
        return frozenlist(super(frozenlist, self).__rmul__(other))

    def __repr__(self):
        return frozenlist.__name__ + '(' + super(frozenlist, self).__repr__() + ')'

    __setattr__ = __delattr__ = _attribute_error
    __setitem__ = __delitem__ = __setslice__ = __delslice__ = __iadd__ = __imul__ = _mutability_error
    append = extend = insert = pop = remove = reverse = sort = _mutability_error

class frozendict(_rst_clean_dict):
    """
    Creates a specialization of dict with the features that it cannot be
    modified and it ensures that all its contained items are immutable.

    A frozendict is created and used like a dict.  It can always be used as a
    dictionary key or as a set member.  Unlike a dict, attempts to modify the
    object will raise a TypeError or AttributeError, and the constructor will
    raise a TypeError if any of the (recursively) contained items are mutable,
    i.e. cannot be hashed.

    >>> d = dict((('a', 1), ('b', 2)))
    >>> f = frozendict(d)
    >>> f
    frozendict({'a': 1, 'b': 2})
    >>> f['a']
    1

    >>> dict(f)
    {'a': 1, 'b': 2}

    Copy construction is cheap

    >>> frozendict(f) is f
    True
    
    Can't put a dict into a set

    >>> s = set()
    >>> s.add(d)  #doctest: +ELLIPSIS
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: ...unhashable...

    Can put a frozendict into a set

    >>> s.add(frozendict(d))
    >>> s
    set([frozendict({'a': 1, 'b': 2})])

    Each item in the constructor iterable must be immutable

    >>> frozendict((('a', 1), ('b', 2), ('c', ['mutable item is bad'])))  #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
        frozendict((('a', 1), ('b', 2), ('c', ['mutable item is bad'])))
      File "<stdin>", line ###, in __init__
      File "<stdin>", line ###, in <genexpr>
    TypeError: ...unhashable...
    """
    __slots__ = ('_hash',)
    def __new__(cls, arg=()):
        return arg if type(arg) is frozendict else super(frozendict, cls).__new__(cls, arg)
    def __init__(self, arg=()):
        if not hasattr(self, '_hash'):
            super(frozendict, self).__init__(arg)
            hashaccum = reduce(int.__xor__, (hash(item) for item in self.iteritems()), 0x00)
            object.__setattr__(self, '_hash', hashaccum)        
    def __hash__(self):
        return self._hash

    def copy(self):
        return self

    # note: we get fromkeys() for free because it's a class method of dict!

    def __repr__(self):
        return frozendict.__name__ + '(' + super(frozendict, self).__repr__() + ')'

    __setattr__ = __delattr__ = _attribute_error
    __setitem__ = __delitem__ = _mutability_error
    clear = pop = popitem = setdefault = update = _mutability_error


class dict_of(_rst_clean_dict):
    """
    A dict specialization that maintians its items as instances of the container
    type that is the first argument to the constructor.  It will automatically
    insert an empty instance of container when a missing key is used.
    Construction from an existing dict, as second constructor arument, will
    automatically convert items to a new instance of container.  That is, copy
    construction goes one level deeper than ordinary dictionary copy
    construction.

    >>> d = dict_of(set)
    >>> d['a'].add(1)
    >>> d['a'].add(2)
    >>> len(d), len(d['a'])
    (1, 2)
    >>> d['a'].add(2)
    >>> len(d), len(d['a'])
    (1, 2)
    >>> d['b'].update(xrange(10, 15))
    >>> d[0] = set((1, 2, 3))
    >>> d
    dict_of(set, {'a': set([1, 2]), 0: set([1, 2, 3]), 'b': set([10, 11, 12, 13, 14])})
    >>> #dict_of(set, frozendict_of(frozenset, d))
    dict_of(set, {'a': set([1, 2]), 0: set([1, 2, 3]), 'b': set([10, 11, 12, 13, 14])})
    
    Whenever you directly set an item at a key in the dictionary, the value you
    provide will be iterated over during the copy into the dict's new container
    for that key.  Thus, you can assign directly from a generator:

    >>> d[10] = (x*x for x in xrange(-4, 4))
    >>> d[10]
    set([16, 9, 4, 0, 1])

    But, this means you must be careful when you want to set the container for a
    key in the dict with a single new value.  In this case you must make sure
    that the provided value is wrapped in a container.  Failure to do so gives
    unintended results, and silently so if the value is itself an iterable
    object.  E.g., when trying to make the item at key 0 be a set containing a
    single string, a simple assignment will not behave as intended:

    >>> d[0] = 'abc'
    >>> d[0]
    set(['a', 'c', 'b'])

    This rarely used, but effective, syntax will wrap the item in a tuple:

    >>> d[0] = 'abc',
    >>> d[0]
    set(['abc'])
    
    Or use a list:

    >>> d[1] = ['def']
    >>> d[1]
    set(['def'])

    If a non-iterable value is used in an incorrect assignment, you'll get a
    some kind of error regarding the object:

    >>> d[0] = 10
    Traceback (most recent call last):
      ...
    TypeError: 'int' object is not iterable

    In addition to normal dict iterations, you can iterate through a 'flattened'
    view of the data in the dictionary.  The first item in each yielded tuple is
    a key in the dict, and the second item is one of the values in the container
    at that key:

    >>> for pair in d.iter_flat:
    ...   print pair
    ('a', 1)
    ('a', 2)
    (0, 'abc')
    (10, 16)
    (10, 9)
    (10, 4)
    (10, 0)
    (10, 1)
    ('b', 10)
    ('b', 11)
    ('b', 12)
    ('b', 13)
    ('b', 14)
    (1, 'def')

    """
    def __init__(self, container_type, arg=()):
        self._container_type = container_type
        iterable = arg.iteritems() if isinstance(arg, dict) else iter(arg)
        set_item = super(dict_of, self).__setitem__
        for key, value in iterable:
            set_item(key, container_type(value))
    def __missing__(self, key):
        self[key] = ()
        return self[key]
    def __setitem__(self, key, value):
        super(dict_of, self).__setitem__(key, self._container_type(value))
    def setdefault(self, key, value):
        return super(dict_of, self).setdefault(key, self._container_type(value))
    def __repr__(self):
        return dict_of.__name__ + '(' + self._container_type.__name__ + ', ' + super(dict_of, self).__repr__() + ')'
    @property
    def iter_flat(self):
        return ((key, value_item) for key, value in self.iteritems() for value_item in value)


class dict_of_set(_rst_clean_dict):
    """
    A dict specialization that requires its items to be instances of set.  Will
    automatically insert an empty set when a missing key is used.  Construction
    will convert a frozendict_of_set to a dict_of_set.

    >>> d = dict_of_set()
    >>> d['a'].add(1)
    >>> d['a'].add(2)
    >>> len(d), len(d['a'])
    (1, 2)
    >>> d['a'].add(2)
    >>> len(d), len(d['a'])
    (1, 2)
    >>> d['b'].update(xrange(10, 15))
    >>> d[0] = set((1, 2, 3))
    >>> d
    dict_of_set({'a': set([1, 2]), 0: set([1, 2, 3]), 'b': set([10, 11, 12, 13, 14])})
    >>> dict_of_set(frozendict_of_set(d))
    dict_of_set({'a': set([1, 2]), 0: set([1, 2, 3]), 'b': set([10, 11, 12, 13, 14])})
    
    >>> d[0] = 'abc'
    Traceback (most recent call last):
      ...
    TypeError: for key 0, expected a set object, got a str
    """
    @staticmethod
    def _check_set(key, value):
        if not (isinstance(value, set)):
            raise TypeError("for key %r, expected a set object, got a %s" % (key, type(value).__name__,))
    def __init__(self, arg=()):
        set_item = self.__setitem__
        for key, value in dict(arg).iteritems():
            set_item(key, set(value) if isinstance(value, frozenset) else value)
        self._verify()
    def _verify(self):
        check_set = self._check_set
        for key, value in self.iteritems():
            check_set(key, value)
    def __missing__(self, key):
        result = self[key] = set()
        return result
    def __setitem__(self, key, value):
        self._check_set(key, value)
        super(dict_of_set, self).__setitem__(key, value)
    def setdefault(self, key, value):
        self._check_set(key, value)
        return super(dict_of_set, self).setdefault(key, value)

    def __repr__(self):
        return dict_of_set.__name__ + '(' + super(dict_of_set, self).__repr__() + ')'
        
class frozendict_of_set(_rst_clean_dict):
    """
    A dict specialization that is immutable and that requires its items to be
    instances of frozenset.  Construction will convert a dict_of_set to a
    frozendict_of_set.

    >>> d = dict_of_set()
    >>> d['a'].update(xrange(5, 10))
    >>> d['b'].update(xrange(10, 15))
    >>> f = frozendict_of_set(d)
    >>> f
    frozendict_of_set({'a': frozenset([8, 9, 5, 6, 7]), 'b': frozenset([10, 11, 12, 13, 14])})
    >>> f[0] = frozenset((1, 2, 3))
    Traceback (most recent call last):
      ...
    TypeError: 'frozendict_of_set' object does not support being modified
    """
    __slots__ = ('_hash',)
    @staticmethod
    def _check_frozenset(key, value):
        if not (isinstance(value, frozenset)):
            raise TypeError("for key %r, expected a frozenset object, got a %s" % (key, type(value).__name__,))
    def __new__(cls, arg=()):
        return arg if type(arg) is frozendict_of_set else super(frozendict_of_set, cls).__new__(cls, arg)
    def __init__(self, arg=()):
        if not hasattr(self, '_hash'):
            check_frozenset = self._check_frozenset
            set_item = super(frozendict_of_set, self).__setitem__
            for key, value in dict(arg).iteritems():
                value = frozenset(value) if isinstance(value, set) else value
                check_frozenset(key, value)
                set_item(key, value)
            hashaccum = reduce(int.__xor__, (hash(item) for item in self.iteritems()), 0x00)
            object.__setattr__(self, '_hash', hashaccum)        
    def _verify(self):
        check_frozenset = self._check_frozenset
        for key, value in self.iteritems():
            check_frozenset(key, value)
    def __hash__(self):
        return self._hash

    def copy(self):
        return self

    # note: we get fromkeys() for free because it's a class method of dict!

    def __repr__(self):
        return frozendict_of_set.__name__ + '(' + super(frozendict_of_set, self).__repr__() + ')'

    __setattr__ = __delattr__ = _attribute_error
    __setitem__ = __delitem__ = _mutability_error
    clear = pop = popitem = setdefault = update = _mutability_error

StrictError = type('StrictError', (ValueError,), {})
def izipstrict(*args):
    """
    XXX broken!  this function does not always raise the error it claims to
    implement; it needs a layer of counters to see how much was used from each
    iterable....

    Returns a generator very like the itertools izip function.  The arguments
    are treated like arguments to izip.  Unlike izip this function ensures that
    all the iterators finished after the same number of items.  It raises a
    StrictError if not all the ierators have finished.

    If StrictError is raised, in addition to a human-readable message, the
    exception will have attributes 'finished', 'unfinished' and 'values', all
    tuples corresponding respectively to the indices in the argument list of the
    iterators that had finished, the indices of the iterators that had not
    finished, and the initial yields from those unfinished iterators.  Note that
    process of getting each initial yield can itself finish iterators in the
    unfinished set.

    >>> tuple(izipstrict(xrange(5), xrange(5)))
    ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4))

    Uncaught error

    >>> tuple(izipstrict(xrange(5), xrange(4), xrange(1,6), xrange(4)))
    Traceback (most recent call last):
       ...
    StrictError: out of 4 iterators, 2 were unfinished: at argument indices (0, 2)

    Caught error and attributes

    >>> try: tuple(izipstrict(xrange(5), xrange(4), xrange(1,6), xrange(4)))
    ... except StrictError, e: print e.finished, e.unfinished, e.values
    (1, 3) (0, 2) (4, 5)
    """
    nexts = tuple(iter(arg).next for arg in args)
    finished = list()
    build = list()
    build_append = build.append
    while True:
        del build[:]
        for index, next in enumerate(nexts):
            try:
                build_append(next())
            except StopIteration:
                finished.append(index)
        if finished and build:
            unfinished = tuple(sorted(frozenset(xrange(len(nexts))) - frozenset(finished)))
            assert len(unfinished) == len(build)
            err = StrictError("out of %d iterators, %d were unfinished: at argument indices %s"
                              % (len(nexts), len(unfinished), tuple(unfinished)))
            err.finished = tuple(finished)
            err.unfinished = unfinished
            err.values = tuple(build)
            raise err
        if build:
            yield tuple(build)
        else:
            assert len(finished) == len(nexts)
            raise StopIteration
        

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
