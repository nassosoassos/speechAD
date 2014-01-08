###########################################################################
#
# File:         yamldata.py (directory: ./py/onyx/textdata)
# Date:         21-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Wrapper around Yaml data
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 - 2009 The Johns Hopkins University
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
    Tools for decoding and encoding Onyx text-based data into YAML documents.

    A text-based Onyx object is a Yaml sequence of a header and a sequence of
    lines.  The header is a map that describes the type and version of the data.
    The sequence of lines is the content of the data.  Each line is a separate
    record of white-space separated tokens.  The reader, YamldataReader, parses
    the stream to get the header and the data.  The YamldataReader is intended
    to be used in an iteration context in order to yield a list of the tokens on
    each line.

    An example

    >>> doc = '''
    ... ---
    ... - __onyx_yaml__meta_version : '1'
    ...   __onyx_yaml__stream_type : IndexedObjectSet
    ...   __onyx_yaml__stream_version : '0'
    ...   __onyx_yaml__stream_options : implicit_index=True
    ...   
    ... -
    ...   # format for IndexedObjectSet, where presence of index field depends on value of implicit_index
    ...   # [index] module factory args-string
    ...   - onyx.signalprocessing.spectrum PreEmphasis  3dB=1500*hz
    ...   - onyx.signalprocessing.window Sliding  length=25000*usec  shift=10000*usec
    ... ...
    ...   '''

    Examples loading a single document from a string or a stream

    Create a reader for this data

    >>> reader = YamldataReader(doc, stream_type='IndexedObjectSet', stream_version='0')
    
    >>> x = list(reader)
    >>> x
    [['onyx.signalprocessing.spectrum', 'PreEmphasis', '3dB=1500*hz'], ['onyx.signalprocessing.window', 'Sliding', 'length=25000*usec', 'shift=10000*usec']]

    >>> x == list(YamldataReader(cStringIO.StringIO(doc), stream_type='IndexedObjectSet', stream_version='0'))
    True

    >>> sorted(reader.keys())
    ['current_line_contents', 'current_line_number', 'meta_version', 'stream_options', 'stream_type', 'stream_version']

    >>> reader.current_line_number
    7
    >>> reader.current_line_contents
    'onyx.signalprocessing.window Sliding  length=25000*usec  shift=10000*usec'
    
    >>> reader
    attrdict({'current_line_contents': 'onyx.signalprocessing.window Sliding  length=25000*usec  shift=10000*usec', 'stream_options': 'implicit_index=True', 'stream_version': '0', 'stream_type': 'IndexedObjectSet', 'current_line_number': 7, 'meta_version': '1'})

    Example of a string or stream with two documents
    
    >>> doc2 = doc + '''
    ...
    ... ---
    ... - __onyx_yaml__meta_version : '1'
    ...   __onyx_yaml__stream_type : IndexedObjectSet
    ...   __onyx_yaml__stream_version : '0'
    ...   __onyx_yaml__stream_options : implicit_index=True
    ...   
    ... -
    ...   # format for IndexedObjectSet, where presence of index field depends on value of implicit_index
    ...   # [index] module factory args-string
    ...   - onyx.signalprocessing.spectrum PreEmphasis  3dB=2500*hz
    ...   - onyx.signalprocessing.window Sliding  length=25000*usec  shift=10000*usec
    ... ...
    ...   '''

    A plain old YamldataReader only returns one document, the first in the
    string or stream

    >>> list(YamldataReader(doc2, stream_type='IndexedObjectSet', stream_version='0'))
    [['onyx.signalprocessing.spectrum', 'PreEmphasis', '3dB=1500*hz'], ['onyx.signalprocessing.window', 'Sliding', 'length=25000*usec', 'shift=10000*usec']]


    To read multiple documents from the string or stream, make a
    YamldataGenerator

    >>> docgen = YamldataGenerator(doc2)

    Then, each YamldataReader instance gets the next document.  So,
    get the first document

    >>> x = YamldataReader(docgen, stream_type='IndexedObjectSet', stream_version='0')
    >>> list(x)
    [['onyx.signalprocessing.spectrum', 'PreEmphasis', '3dB=1500*hz'], ['onyx.signalprocessing.window', 'Sliding', 'length=25000*usec', 'shift=10000*usec']]

    Get the next document

    >>> y = YamldataReader(docgen, stream_type='IndexedObjectSet', stream_version='0')
    >>> list(y)
    [['onyx.signalprocessing.spectrum', 'PreEmphasis', '3dB=2500*hz'], ['onyx.signalprocessing.window', 'Sliding', 'length=25000*usec', 'shift=10000*usec']]

    For documentation on YamldataReader errors and what causes them, see help(yamldata_reader_errors).
"""

from __future__ import with_statement

from itertools import izip as _izip, repeat as _repeat
import cStringIO

from onyx.builtin import attrdict
import syck as _syck

class YamldataGenerator(object):
    def __init__(self, instream):
        # note: this causes Syck to parse the entire stream
        self._next = _syck.load_documents(instream).next
        if hasattr(instream, 'name'):
            self.current_filename = instream.name


    def __iter__(self):
        return self
    def next(self):
        return self._next()


class YamldataBase(attrdict):
    # configurable aspects of YamldataBase
    HEADER_NAME_PREFIX = '__onyx_yaml__'
    VERSION = '1'

    required_headers = frozenset(('meta_version', 'stream_type', 'stream_version',))
    optional_headers = frozenset(('stream_options',))

    # derived aspects of YamldataBase
    assert not (required_headers & optional_headers)
    valid_headers = required_headers ^ optional_headers
    # print valid_headers

    @staticmethod
    def prefix_header_name(header):
            return YamldataBase.HEADER_NAME_PREFIX + header

    @staticmethod
    def prefix_header_names(headers):
        for header in headers:
            yield YamldataBase.prefix_header_name(header)

    @staticmethod
    def build_header_dict(stream_type, stream_version, stream_options=None):
        temp = dict()
        temp['meta_version'] = YamldataBase.VERSION
        temp['stream_type'] = stream_type
        temp['stream_version'] = stream_version
        if stream_options is not None:
            temp['data_option'] = data_option
        check = frozenset(temp.keys())
        assert check <= YamldataBase.valid_headers
        # Now build dict with prefixed names
        return dict([(YamldataBase.prefix_header_name(key), value) for (key,value) in temp.items()])


class YamldataReader(YamldataBase):
    """
    This object is an attrdict for access to the header fields of the Yamldata.
    It is also a one-shot iterator over the contents of the Yamldata, if any,
    where each yield is a list of the white-space separated tokens on the next
    non-empty item in the Yamldata.
    """
    __slots__ = ('_next',)
    def __init__(self, instream,
                 stream_type=None,
                 stream_version=None,
                 no_stream_options=False,
                 header_only=False):

        if not isinstance(instream, YamldataGenerator):
            instream = YamldataGenerator(instream)
        have_filename = hasattr(instream, 'current_filename')
        if have_filename:
            self.current_filename = instream.current_filename
        stream_name = (self.current_filename if have_filename else "Yamldata stream")

        doc = instream.next()
        if (not hasattr(doc, '__len__')) or len(doc) == 0:
            raise ValueError("no header found in %s; are you sure this is a Yamldata source?" 
                             % stream_name)

        # attrdict of all the Onyx headers
        try:
            header = attrdict(doc[0])
        except:
            raise ValueError("bad header structure in %s: [%s] - are you sure this is a Yamldata source?"
                             % (stream_name, doc[0]))
            
        if len(doc) > 2:
            raise ValueError("bad document structure in %s, expected 1 or 2 sub-parts, got %d"
                             % (stream_name, len(doc)))
        # print doc

        if header_only:
            if len(doc) != 1:
                raise ValueError("bad document structure in %s, expected only a header"
                                 % stream_name)
            data = None
        else:
            if len(doc) != 2:
                raise ValueError("bad document structure in %s, expected both a header and a body"
                             % stream_name)
            data = doc[1]

        missing_headers = frozenset(self.prefix_header_names(self.required_headers)) - frozenset(header)
        if missing_headers:
            raise ValueError("missing the following required headers in %s: %s"
                             % (stream_name, (' '.join(repr(header) for header in sorted(missing_headers)))))

        invalid_headers = frozenset(header) - frozenset(self.prefix_header_names(self.valid_headers))
        if invalid_headers:
            raise ValueError("unexpected headers in %s: %s"
                             % (stream_name, (' '.join(repr(header) for header in sorted(invalid_headers)))))

        for base_name, prefixed_name in _izip(self.valid_headers, self.prefix_header_names(self.valid_headers)):
            if prefixed_name in header:
                self[base_name] = header[prefixed_name]

        # check the fields
        if self.meta_version != self.VERSION:
            raise ValueError("unexpected meta_version in %s: expected %s, got %s"
                             % (stream_name, self.VERSION, self.meta_version))
        if stream_type is not None and self.stream_type != stream_type:
            raise ValueError("unexpected stream_type in %s: expected %r, got %r"
                             % (stream_name, stream_type, self.stream_type))
        if stream_version is not None and self.stream_version != stream_version:
            raise ValueError("unexpected stream_version in %s: expected %s, got %s"
                             % (stream_name, repr(stream_version), repr(self.stream_version)))

        if no_stream_options and self.hasattr.stream_options:
            raise ValueError("unexpected presence of stream_options in header: %r" % (self.stream_options,))

        self.current_line_number = 4 + (0 if no_stream_options else 1)
        def itr():
            for line in data:
                self.current_line_contents = line
                self.current_line_number += 1
                # Note: PyYAML will implicitly convert tokens which
                # match certain regexps to their "natural" types.  The
                # effect is that if a line has only a single token
                # which can be converted to float or int, it will be
                # so converted.  Here we detect that and convert back
                # to a tuple with one string to make our output consistent.
                if type(line) != str:
                    parts = (str(line),)
                else:
                    parts = line.split()
                if not parts:
                    continue
                yield parts
        self._next = itr().next

    def __iter__(self):
        return self
    
    def next(self):
        return self._next()


class YamldataWriter(YamldataBase):
    """
    >>> stream0 = cStringIO.StringIO()
    >>> yw0 = YamldataWriter(stream0, 'MyType', '0')
    >>> src = (('var0', 0), ('var1', 1))
    >>> yw0.write_document(src)

    >>> print stream0.getvalue()
    ---
    - __onyx_yaml__stream_version: "0"
      __onyx_yaml__meta_version: "1"
      __onyx_yaml__stream_type: "MyType"
    - - var0 0
      - var1 1
    <BLANKLINE>

    >>> stream0.seek(0)
    >>> yr = YamldataReader(stream0)
    >>> body = list(yr)

    >>> stream1 = cStringIO.StringIO()
    >>> yw1 = YamldataWriter(stream1, 'MyType', '0')
    >>> yw1.write_document(body)
    >>> stream1.getvalue() == stream0.getvalue()
    True

    Add a few more documents to this stream

    >>> yw1.write_document(body)
    >>> yw1.write_document(body)
    >>> print stream1.getvalue()
    ---
    - __onyx_yaml__stream_version: "0"
      __onyx_yaml__meta_version: "1"
      __onyx_yaml__stream_type: "MyType"
    - - var0 0
      - var1 1
    ---
    - __onyx_yaml__stream_version: "0"
      __onyx_yaml__meta_version: "1"
      __onyx_yaml__stream_type: "MyType"
    - - var0 0
      - var1 1
    ---
    - __onyx_yaml__stream_version: "0"
      __onyx_yaml__meta_version: "1"
      __onyx_yaml__stream_type: "MyType"
    - - var0 0
      - var1 1
    <BLANKLINE>
    
    """
    def __init__(self, outstream, stream_type, stream_version, stream_options=None):
        self._header = YamldataBase.build_header_dict(stream_type, stream_version, stream_options)
        self._outstream = outstream

    def write_document(self, iterable):
        """
        Write a single Yamldata document to the outstream.  iterable should produce sequences which
        will each be converted to one line of space-separated tokens.  This function may be called
        multiple times to produce streams with multiple documents.
        """
        body = [' '.join([str(token) for token in seq]) for seq in iterable]
        # I think using syck here would be fine, but it has a bug that causes it
        # to write bad data, so it's currently unusable.  There's a fix to this
        # problem available now, but we have our own output routine that seems
        # to do what we need.
        ### _syck.dump((self._header, body), self._outstream, explicit_typing=False)
        # So we do it ourselves:
        self.output_as_yaml_doc(body)


    def output_as_yaml_doc(self, body):
        # Write document start 
        DOC_MARKER = "---"
        self._outstream.write(DOC_MARKER + "\n")
        ITEM_MARKER = "- "
        INDENT = " " * len(ITEM_MARKER)
        
        # Write document header 
        for i, (k,v) in enumerate(self._header.items()):
            out = (ITEM_MARKER if i == 0 else INDENT) + str(k) + ': "' + str(v) + '"\n'
            self._outstream.write(out)

        # Write body
        for i, line in enumerate(body):
            out = (ITEM_MARKER if i == 0 else INDENT) + ITEM_MARKER + line + "\n"
            self._outstream.write(out)

# This function exists to have a doctest string that gets run without cluttering
# up a real doc string with all these tests and to be helpful documentation.  
def yamldata_reader_errors():
    """
    >>> YamldataReader("")
    Traceback (most recent call last):
    ...
    StopIteration

    >>> YamldataReader("1")
    Traceback (most recent call last):
    ...
    ValueError: no header found in Yamldata stream; are you sure this is a Yamldata source?

    >>> YamldataReader("a")
    Traceback (most recent call last):
    ...
    ValueError: bad header structure in Yamldata stream: [a] - are you sure this is a Yamldata source?

    >>> YamldataReader("- __onyx_yaml__meta_version : '1' \\n- - foo\\n- - bar ")
    Traceback (most recent call last):
    ...
    ValueError: bad document structure in Yamldata stream, expected 1 or 2 sub-parts, got 3

    >>> YamldataReader("- __onyx_yaml__meta_version : '1' \\n- - foo", header_only=True)
    Traceback (most recent call last):
    ...
    ValueError: bad document structure in Yamldata stream, expected only a header

    >>> YamldataReader("- __onyx_yaml__meta_version : '1'", header_only=False)
    Traceback (most recent call last):
    ...
    ValueError: bad document structure in Yamldata stream, expected both a header and a body

    Some error cases where the well-formatted data doesn't match the
    code's expectations:

    >>> doc = '''
    ... ---
    ... - __onyx_yaml__meta_version : '1'
    ...   __onyx_yaml__stream_type : IndexedObjectSet
    ...   __onyx_yaml__stream_version : '0'
    ...   __onyx_yaml__stream_options : implicit_index=True
    ...   
    ... -
    ...   # format for IndexedObjectSet, where presence of index field depends on value of implicit_index
    ...   # [index] module factory args-string
    ...   - onyx.signalprocessing.spectrum PreEmphasis  3dB=1500*hz
    ...   - onyx.signalprocessing.window Sliding  length=25000*usec  shift=10000*usec
    ... ...
    ...   '''

    >>> reader = YamldataReader(doc, stream_type='BigObject', stream_version='0')
    Traceback (most recent call last):
       ...
    ValueError: unexpected stream_type in Yamldata stream: expected 'BigObject', got 'IndexedObjectSet'

    >>> reader = YamldataReader(doc, stream_type='IndexedObjectSet', stream_version='0.1')
    Traceback (most recent call last):
       ...
    ValueError: unexpected stream_version in Yamldata stream: expected '0.1', got '0'
    
    >>> reader = YamldataReader(doc, stream_type='IndexedObjectSet', stream_version='0', no_stream_options=True)
    Traceback (most recent call last):
       ...
    ValueError: unexpected presence of stream_options in header: 'implicit_index=True'
    
    >>> YamldataReader("- __onyx_yaml__meta_version : '0' \\n  __onyx_yaml__stream_type : MyType  \\n  __onyx_yaml__stream_version : '0'  \\n- - foo \\n  - bar")
    Traceback (most recent call last):
       ...
    ValueError: unexpected meta_version in Yamldata stream: expected 1, got 0


    Some structural error cases:

    >>> YamldataReader("- __onyx_yaml__meta_version : '1' \\n- - foo ")
    Traceback (most recent call last):
       ...
    ValueError: missing the following required headers in Yamldata stream: '__onyx_yaml__stream_type' '__onyx_yaml__stream_version'
    
    >>> YamldataReader("- __onyx_yaml__meta_version : '1' \\n  __onyx_yaml__stream_type : MyType  \\n  __onyx_yaml__stream_version : '0'  \\n  __onyx_yaml__bogus_header : bogus header value  \\n- - foo \\n  - bar")
    Traceback (most recent call last):
       ...
    ValueError: unexpected headers in Yamldata stream: '__onyx_yaml__bogus_header'

    This concludes testing for this module
    """
            
if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
