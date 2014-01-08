###########################################################################
#
# File:         onyxtext.py
# Date:         Mon 25 Aug 2008 12:37
# Author:       Ken Basye
# Description:  A text format for line-oriented serialization of memory structures
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008 The Johns Hopkins University
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
A text format for line-oriented serialization of memory structures


"""

from numpy import array as numpy_array
from onyx.util.floatutils import float_to_readable_string, readable_string_to_float
from onyx.util.singleton import Singleton
from itertools import chain, repeat


# XXX Still need a better name for these classes.
class OnyxTextBase(object):
    STREAM_TYPE = "OnyxText"
    STREAM_VERSION = "0"
    _HEADER_FORMAT_STRING = "stream_type " + STREAM_TYPE + " stream_version " + STREAM_VERSION + "  data_type %s  data_version %s"
    _HEADER_TOKEN_COUNT = len((_HEADER_FORMAT_STRING % ('dummy', '0')).split())
    _IC_TOKEN = "IndexedCollection"
    _LIST_TOKEN = "List"
    _ARRAY_TOKEN = "Array"
    _SINGLETON_TOKEN = "Singleton"

    def _check_header_tokens(self, tokens, data_type, data_version):
        expected = (self._HEADER_FORMAT_STRING % (data_type, data_version)).split()
        self._verify_token_count(len(expected), tokens, "%s header specification" % (self.STREAM_TYPE,))
        # Check static tokens
        for i in (0, 2, 4, 6):
            self._verify_thing(expected[i], tokens[i], "header token")
        # Now check things that could actually vary
        self._verify_thing(expected[1], tokens[1], "stream type")
        self._verify_thing(expected[3], tokens[3], "stream version")
        self._verify_thing(data_type, tokens[5], "data type")
        self._verify_thing(data_version, tokens[7], "data version")
        

    @classmethod
    def _make_header_tokens(cls, data_type, data_version):
        tokens = (cls._HEADER_FORMAT_STRING % (data_type, data_version)).split()
        if len(tokens) != cls._HEADER_TOKEN_COUNT:
            raise ValueError("data_type and data_version must be single tokens (no spaces)")
        return tuple(tokens)


OnyxTextReaderParseError = type("OnyxTextReaderParseError", (StandardError,), dict())

class OnyxTextReader(OnyxTextBase):
    """
    This class supports parsing from a stream which produces sequences of string tokens.  The intent
    is to support a particular style of line-oriented text file or stream which is tokenized by some
    lower level.  Five types of representations are supported: simple scalars, lists, arrays, and
    indexed collections.  Each representation type uses a different syntax, as described in the
    various read_XXX functions.  See help on read_scalar, read_singleton, read_list, read_array, and
    read_indexed_collection for the details on each representation syntax.  Also see help on
    OnyxTextWriter for the details on how to generate these representations.

    The iterable argument to the constructor should be a source of sequences to be parsed, the first
    of which must be a valid OnyxText header sequence.  If the data_type and/or date_version
    arguments are not None, they will be checked against the corresponding values in the header
    sequence.

    Generically, the read_XXX functions on the stream object take information which is meant to be
    verified and/or used in parsing.  E.g., specifying name='foo' means the name of the parsed
    entity will be verified to be 'foo' and not any other name.  The return values are the object
    name and the object.

    >>> src = (OnyxTextReader._make_header_tokens('test', '0'), ('name0', '4'), ('name1', '3.14159'))
    >>> src[0]
    ('stream_type', 'OnyxText', 'stream_version', '0', 'data_type', 'test', 'data_version', '0')
    
    >>> ctr0 = OnyxTextReader(src, data_type='test', data_version='0')
    >>> ctr0.data_type
    'test'
    >>> ctr0.data_version
    '0'
    >>> ctr0.read_scalar(name='name0', rtype=int)
    ('name0', 4)
    >>> ctr0.read_scalar(name='name0', rtype=int)
    Traceback (most recent call last):
    ...
    OnyxTextReaderParseError: Expected to read name name0, but read name1
    """

    def __init__(self, iterable, data_type=None, data_version=None):
        self._stream = iter(iterable)
        # Read and check header
        tokens = self._stream.next()
        self._check_header_tokens(tokens, data_type, data_version)
        self._data_type = data_type
        self._data_version = data_version

    @property
    def data_type(self):
        return self._data_type

    @property
    def data_version(self):
        return self._data_version

    def read_scalar(self, name=None, rtype=str):
        """
        A scalar is represented on one line with two tokens, the name of the scalar and the value.
        If name is not none, it will be checked against the name of the scalar.  The value will be
        converted to rtype before return.
        
        >>> src = (OnyxTextReader._make_header_tokens('test', '0'), ('name0', '4'), ('name1', '3.14159'))

        >>> ctr0 = OnyxTextReader(src)
        >>> ctr0.read_scalar(rtype=int)
        ('name0', 4)
        >>> n,v = ctr0.read_scalar(rtype=float)
        >>> n, float_to_readable_string(v)
        ('name1', '+(+0001)0x921f9f01b866e')

        >>> ctr0 = OnyxTextReader(src)
        >>> ctr0.read_scalar('name0', int)
        ('name0', 4)
        >>> ctr0.read_scalar('name1')
        ('name1', '3.14159')
        """
        tokens = self._next_line()
        self._verify_token_count(2, tokens, "scalar specification line")
        (rname, val) = tokens
        self._verify_thing(name, rname, 'name')
        return rname, rtype(val)


    def read_singleton(self, name=None):
        """
        A singleton is represented on one line with three tokens: the fixed token 'Singleton', the
        name of the variable, and the name of the singleton.  If name is not none, it will be
        checked against the name of the variable. The return value will be an object of
        onyx.util.singleton.Singleton.
        
        >>> src = (OnyxTextReader._make_header_tokens('test', '0'), ('test', 'Singleton', 'onyx.textdata.onyxtext.test'))

        >>> ctr0 = OnyxTextReader(iter(src))
        >>> n, s = ctr0.read_singleton('test')
        >>> s
        onyx.util.singleton.Singleton('onyx.textdata.onyxtext.test')

        """
        tokens = self._next_line()
        self._verify_token_count(3, tokens, "singleton specification line")
        (rname, singleton_token, singleton_name) = tokens
        self._verify_thing(self._SINGLETON_TOKEN, singleton_token, 'token')
        self._verify_thing(name, rname, 'name')
        return rname, Singleton(singleton_name)


    def read_list(self, name=None, rtype=str, count=None):
        """
        A list is represented as a specification on one line and the list items on the next line.
        The specification consists of the name of the list, the token 'List', and the number of
        items.  If name is not none, it will be checked against the name of the list. The tokens on
        the second line will be converted to rtype before return.

        >>> src = (OnyxTextReader._make_header_tokens('test', '0'), ('name0', 'List', '4'),(0,1,4,9))
        >>> ctr0 = OnyxTextReader(src)
        >>> ctr0.read_list('name0', int)
        ('name0', [0, 1, 4, 9])

        >>> src = (OnyxTextReader._make_header_tokens('test', '0'), ('name1', 'List', '1'),("149",))
        >>> ctr0 = OnyxTextReader(src)
        >>> ctr0.read_list('name1', int, count=1)
        ('name1', [149])
        """
        tokens = self._next_line()
        self._verify_token_count(3, tokens, "list specification line")
        (rname, list_token, exp_len) = tokens
        exp_len = int(exp_len)
        self._verify_thing(self._LIST_TOKEN, list_token, 'token')
        self._verify_thing(name, rname, 'name')
        self._verify_thing(count, exp_len, 'list element count')
        elts = self._next_line()
        self._verify_token_count(exp_len, elts, "list contents line")
        return rname, [rtype(t) for t in elts]

    def read_array(self, name=None, rtype=str, dim=None, shape=None):
        """
        An array is represented as a specification on one line and the array contents on subsequent
        lines.  The specification consists of the name of the array, the token 'Array', the
        dimension of the array, and the sizes of the dimensions.  Each subsequent line must contain
        as many elements as the size of the last dimension, and there must be enough lines to
        complete the array.  Note: this layout is consistent with how Numpy prints arrays.  vtype is
        used to construct and array of the correct type.  If name is not none, it will be checked
        against the name of the array.  If dim is not None, it will be checked against the specified
        dimension of the array.  If shape is not None, it must be a tuple of ints and will be
        checked against the specified shape of the array.
        
        >>> src = (OnyxTextReader._make_header_tokens('test', '0'), ('name0', 'Array', 2, 3, 4),(0,1,4,9),(1,2,3,4),(10,20,30,40))
        >>> ctr0 = OnyxTextReader(src)
        >>> ctr0.read_array('name0', int, 2, (3,4))
        ('name0', array([[ 0,  1,  4,  9],
               [ 1,  2,  3,  4],
               [10, 20, 30, 40]]))
        """
        tokens = self._next_line()
        self._verify_token_count_min(4, tokens, "array specification line")
        rname = tokens[0]
        array_token = tokens[1]
        spec_dim = int(tokens[2])
        spec_shape = tuple([int(t) for t in tokens[3:]])
        self._verify_thing(self._ARRAY_TOKEN, array_token, 'token')
        self._verify_thing(name, rname, 'name')
        self._verify_thing(dim, spec_dim, 'array dimension')
        self._verify_thing(shape, spec_shape, 'array shape')

        # Even if we didn't get expected values, we can verify a few things in the spec.
        if spec_dim < 1:
            self.raise_parsing_error("found array dimension %d - it must be at least 1" % (d,))
        if len(spec_shape) != spec_dim:
            self.raise_parsing_error("array dimension is %d but shape is %s" % (spec_dim, spec_shape))
        n_items_per_line = spec_shape[-1]
        if spec_dim == 1:
            n_lines = 1
        else:
            n_lines = reduce(lambda x,y: x*y, shape[:-1])
        values = []

        if rtype == float:
            unformatter = readable_string_to_float
        else:
            unformatter = lambda x: x
        for i in xrange(n_lines):
            tokens = self._next_line()
            self._verify_token_count(n_items_per_line, tokens, "array values line")
            values += [rtype(unformatter(t)) for t in tokens]
        result = numpy_array(values, dtype=rtype).reshape(shape)
        return rname, result
            

    def read_indexed_collection(self, read_fn, user_data, name=None, header_token_count=None):
        """
        An indexed collection is represented as a specification on one line and the indexed items on
        subsequent lines.  The specification consists of the name of the collection, the token
        'IndexedCollection', the name of the items, and the number of items.  read_fn is a callable
        which takes three arguments, the stream, the user_data and the sequence of header tokens,
        and returns the object read.  Each item is represented as one line consisting of the object
        name, the index, and optional additional fields, followed by as many lines as necessary to
        represent the object (which may be 0).  read_fn will be called once for each object with the
        stream and the tokens on the first line as arguments.  If name is not none, it will be
        checked against the name of the collection.  If header_token_count is not None, it will be checked
        against the number of tokens in each header.

        >>> src = (OnyxTextReader._make_header_tokens('test', '0'), ('collection0', 'IndexedCollection', 'item', '2'),
        ...        ('item', '0', 'hi'), ('X', '4'), ('Y', '4'), ('MyList', 'List', '4'), ('a', 'b', 'c', 'd'),
        ...        ('item', '1', 'there'), ('X', '14'), ('Y', '44'), ('MyList', 'List', '4'), ('foo', 'bar', 'baz', 'foobar'))

        >>> def reader(s, not_used, tokens):
        ...    v,x = s.read_scalar('X', int)
        ...    v,y = s.read_scalar('Y', int)
        ...    v,l = s.read_list('MyList')
        ...    return (int(tokens[1]), tokens[2], x, y, l)

        >>> ctr0 = OnyxTextReader(src)
        >>> ctr0.read_indexed_collection(reader, None, name='collection0', header_token_count=3)
        ('collection0', ((0, 'hi', 4, 4, ['a', 'b', 'c', 'd']), (1, 'there', 14, 44, ['foo', 'bar', 'baz', 'foobar'])))

        """
        tokens = self._next_line()
        self._verify_token_count(4, tokens, "IndexedCollection specification line")
        (rname, ic_token, obj_name, c) = tokens
        self._verify_thing(self._IC_TOKEN, ic_token, "token")
        self._verify_thing(name, rname, 'name')
        num_items = int(c)
        result = []
        for index in xrange(num_items):
            # read the name/index line for this object
            tokens = self._next_line()
            self._verify_token_count_min(2, tokens, "header line")
            self._verify_thing(obj_name, tokens[0], "header line with name")
            self._verify_thing(index, int(tokens[1]), "header line with index")
            if header_token_count is not None:
                self._verify_token_count(header_token_count, tokens, "header line") 
            obj = read_fn(self, user_data, tokens)
            result.append(obj)
        return rname, tuple(result)


    def raise_parsing_error(self, err_str):
        """
        Clients may call this function to raise an error consistent with those raised by the
        read_XXX functions.  If the stream has any of the attributes: current_line_number,
        current_filename, or current_line_contents, additional information will be added to the
        error string.
        """
        line_info = ""
        if hasattr(self._stream, "current_line_number") and self._stream.current_line_number is not None:
            line_info += (" on or near line %d" % self._stream.current_line_number)
        if hasattr(self._stream, "current_filename") and self._stream.current_filename is not None:
            line_info += (" in file %s" % self._stream.current_filename)
        if hasattr(self._stream, "current_line_contents") and self._stream.current_line_contents is not None:
            line_info += ("\n Complete line: [[%s]]" % self._stream.current_line_contents)
        raise(OnyxTextReaderParseError(err_str + line_info))

    def _verify_thing(self, expected, found, what):
        if expected is not None and found != expected:
            self.raise_parsing_error("Expected to read %s %s, but read %s" % (what, expected, found))
    
    def _verify_token_count(self, expected, tokens, what):
        n = len(tokens)
        if n != expected:
            self.raise_parsing_error("Expected %s with %d tokens, but read %d tokens" % (what, expected, n))

    def _verify_token_count_min(self, expected, tokens, what):
        n = len(tokens)
        if n < expected:
            self.raise_parsing_error("Expected %s with at least %d tokens, but read %d tokens" % (what, expected, n))

    def _next_line(self):
        try:
            tokens = self._stream.next()
        except StopIteration:
            self.raise_parsing_error("Unexpected end of stream")
        return tokens


# XXX - so far this class has no members; these could all be static functions
class OnyxTextWriter(OnyxTextBase):
    """
    This class provides generation of sequences of string tokens.  The intent is to support a
    particular style of line-oriented text file or stream which is written out by some lower level.
    Five types of representations are supported: simple scalars, singletons, lists, arrays, and
    indexed collections.  Each representation type uses a different syntax, as described in the
    various gen_XXX functions.  See help on gen_scalar, gen_singleton, gen_list, gen_array, and
    gen_indexed_collection for the details on each representation syntax.  Also see help on
    OnyxTextReader for the details on how to parse these representations.

    Generically, the gen_XXX functions on the writer object take data to be serialized.  The return
    values are generators which will produce the appropriate sequence or sequences for that data.

    >>> writer = OnyxTextWriter()
    >>> header_gen = writer.gen_header('test', '0')
    >>> v1_gen = writer.gen_scalar('var1', 3.14159)
    >>> all_out_gen = chain(header_gen, v1_gen)
    >>> output = tuple(all_out_gen)
    >>> output
    (('stream_type', 'OnyxText', 'stream_version', '0', 'data_type', 'test', 'data_version', '0'), ('var1', '3.14159'))
    
    """
    def gen_header(self, data_type, data_version):
        """
        Generate a OnyxText header tuple with the given data_type and data_version.
        >>> ctw0 = OnyxTextWriter()
        >>> g = ctw0.gen_header('test', '0')
        >>> tuple(g)
        (('stream_type', 'OnyxText', 'stream_version', '0', 'data_type', 'test', 'data_version', '0'),)
        """
        yield self._make_header_tokens(data_type, data_version)

    def gen_scalar(self, name, value):
        """
        >>> ctw0 = OnyxTextWriter()
        >>> g = ctw0.gen_scalar("scalar0", 3.14159)
        >>> tuple(g)
        (('scalar0', '3.14159'),)
        """
        yield (name, str(value))

    def gen_singleton(self, name, value):
        """
        >>> ctw0 = OnyxTextWriter()
        >>> s = Singleton('onyx.textdata.onyxtext.test0')
        >>> g = ctw0.gen_singleton("test0", s)
        >>> tuple(g)
        (('test0', 'Singleton', 'onyx.textdata.onyxtext.test0'),)

        """
        if not isinstance(value, Singleton):
            raise ValueError("Expected instance of type Singleton, got %s" % type(value))
        yield (name, self._SINGLETON_TOKEN, value.name)

    def gen_list(self, name, the_list):
        """
        >>> ctw0 = OnyxTextWriter()
        >>> g = ctw0.gen_list("list0", [1.2, 2.3, 3.4])
        >>> tuple(g)
        (('list0', 'List', '3'), ('1.2', '2.3', '3.4'))
        """
        yield (name, self._LIST_TOKEN, str(len(the_list)))
        yield tuple([str(item) for item in the_list])


    def gen_array(self, name, array):
        """
        >>> arr = numpy_array(range(8), dtype = float).reshape(2,4)
        >>> arr
        array([[ 0.,  1.,  2.,  3.],
               [ 4.,  5.,  6.,  7.]])

        >>> ctw0 = OnyxTextWriter()
        >>> g = ctw0.gen_array("array0", arr)
        >>> tuple(g)
        (('array0', 'Array', '2', '2', '4'), ('+(-1023)0x0000000000000', '+(+0000)0x0000000000000', '+(+0001)0x0000000000000', '+(+0001)0x8000000000000'), ('+(+0002)0x0000000000000', '+(+0002)0x4000000000000', '+(+0002)0x8000000000000', '+(+0002)0xc000000000000'))

        >>> arr = numpy_array(range(8), dtype = float).reshape(2,2,2)
        >>> arr
        array([[[ 0.,  1.],
                [ 2.,  3.]],
        <BLANKLINE>
               [[ 4.,  5.],
                [ 6.,  7.]]])

        >>> ctw0 = OnyxTextWriter()
        >>> g = ctw0.gen_array("array1", arr)
        >>> tuple(g)
        (('array1', 'Array', '3', '2', '2', '2'), ('+(-1023)0x0000000000000', '+(+0000)0x0000000000000'), ('+(+0001)0x0000000000000', '+(+0001)0x8000000000000'), ('+(+0002)0x0000000000000', '+(+0002)0x4000000000000'), ('+(+0002)0x8000000000000', '+(+0002)0xc000000000000'))
        """
        if array.dtype == float:
            formatter = float_to_readable_string
        else:
            formatter = str
        yield (name, self._ARRAY_TOKEN, str(array.ndim)) + tuple([str(s) for s in array.shape])
        if array.ndim == 1:
            n_lines = 1
        else:
            n_lines = reduce(lambda x,y: x*y, array.shape[:-1])
        temp = array.reshape(n_lines, array.shape[-1])
        for line in xrange(n_lines):
            yield tuple([formatter(val) for val in temp[line, :]])


    def gen_indexed_collection(self, name, obj_name, obj_seq, obj_gen):
        """
        >>> def obj_gen(stream, obj):
        ...    return chain((('info', str(len(obj[2]))),),
        ...                 stream.gen_scalar("X", obj[0]),
        ...                 stream.gen_scalar("Y", obj[1]),
        ...                 stream.gen_list("MyList", obj[2]))
        >>> objs = ((3,4,[1,2,3]), (1.2, 2.3, ['a', 'b']))
        >>> ctw0 = OnyxTextWriter()
        >>> g = ctw0.gen_indexed_collection("collection0", "object", objs, obj_gen)
        >>> tuple(g)
        (('collection0', 'IndexedCollection', 'object', 2), ('object', '0', 'info', '3'), ('X', '3'), ('Y', '4'), ('MyList', 'List', '3'), ('1', '2', '3'), ('object', '1', 'info', '2'), ('X', '1.2'), ('Y', '2.3'), ('MyList', 'List', '2'), ('a', 'b'))
        """
        obj_seq = tuple(obj_seq)
        yield (name, self._IC_TOKEN, obj_name, len(obj_seq))
        for i, obj in enumerate(obj_seq):
            gen = obj_gen(self, obj)
            # Yield name/index line - note getting StopIteration here means obj_gen is bad
            yield (obj_name, str(i)) + gen.next()
            # Yield body lines
            for tup in gen:
                yield tup
        
                
if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
