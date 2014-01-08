###########################################################################
#
# File:         textdata.py
# Date:         04-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Support for writing and reading data using the Textdata format
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
Support for writing and reading data using the Textdata format

A format and tool set for encoding line-based records into
white-space-separated tokens and for decoding such records.

>>> True
True
"""

from functools import partial as _partial
from onyx.util.checkutils import check_instance as _check_instance

class TextdataBase(object):
    """
    Base class for Textdata writer and reader

    Holds some basic accessors, constants, and functions; includes some
    invariant assertions
    """

    KEYWORD_PREFIX = '__textdata__'

    VERSION_NAME =           'version'
    VERSION_VALUE =          '1'

    FILE_TYPE_NAME =         'file_type'
    FILE_VERSION_NAME =      'file_version'
    COMMENT_PREFIX_NAME =    'comment_prefix'
    ESCAPE_CHAR_NAME =       'escape_char'

    END_OF_HEADER_NAME =     'end_of_header'
    END_OF_DATA_NAME =       'end_of_data'


    # some ascii characteristics

    SPACE = ' '
    NEWLINE = '\n'
    WHITE = frozenset(SPACE + NEWLINE)

    GRAPHICAL = frozenset('!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')

    # mutually exclusive sets
    assert not (WHITE & GRAPHICAL)

    # legal within comments; and comments are not encoded
    LEGAL_COMMENT = GRAPHICAL | frozenset(SPACE)

    # legal in output stream
    LEGAL = GRAPHICAL | WHITE

    # this restriction is to avoid semantic subtleties of nulls in
    # strings and to prevent abuse of this format for encoding unicode
    # or binary data; unicode and binary encoding, if either is
    # necessary, should be handled by revisions to the specification
    NOT_ENCODEABLE = frozenset('\x00')

    HEX = frozenset('0123456789ABCDEFabcdef')
    EMPTY = ''
    ESCAPED_EMPTY = "''"
    ESCAPED_EMPTY_SET = frozenset(ESCAPED_EMPTY)
    assert not (HEX & ESCAPED_EMPTY_SET)
    NOT_ESCAPE = HEX | ESCAPED_EMPTY_SET

    # characters in the comment_prefix will get excluded as well
    LEGAL_ESCAPE = GRAPHICAL - NOT_ESCAPE

    # accessors and functions

    @property
    def file_type(self):
        """the name of the type of data in the stream or None, a read-only attribute"""
        return self._file_type

    @property
    def file_version(self):
        """the version of the type of data in the stream or None, a read-only attribute"""
        return self._file_version


    def check_configuration(self):

        file_type = self._file_type 
        file_version = self._file_version 
        comment_prefix = self.comment_prefix 
        escape_char = self.escape_char 
        headerless = not self.gets_header

        def verify_graphical(token, token_name, single_char=False):
            if not token or (single_char and (len(token) != 1)) or (frozenset(token) - self.GRAPHICAL):
                or_more, ess = ('', '',) if single_char else (' or more', 's',)
                raise ValueError("expected one%s graphical character%s in %s, got %s" % (or_more, ess, token_name, repr(token),))

        verify_graphical(comment_prefix, 'comment_prefix')
        verify_graphical(escape_char, 'escape_char', True)
        if escape_char in frozenset(comment_prefix):
            raise ValueError("escape_char %s is part of comment_prefix %s" % (repr(escape_char), repr(comment_prefix),))
        if escape_char not in self.LEGAL_ESCAPE:
            raise ValueError("special character %s cannot be the escape_char" % (repr(escape_char),))

        if (file_type is None) != (file_version is None):
            raise ValueError("must specifiy both or neither of file_type and file_version, got %s and %s" % (repr(file_type), repr(file_version),))

        if file_type is not None:
            if headerless:
                raise ValueError("expected no file_type because headerless is %s, but got %s" % (repr(headerless), repr(file_type),))
            verify_graphical(file_type, 'file_type')
            verify_graphical(file_version, 'file_version')


    @staticmethod
    def iter_rawtokens(stream):
        """Generator yields list of raw tokens on each line, skips blank lines"""
        # type-checking and a function all in one
        str_split = str.split
        for line in stream:
            tokens = str_split(line)
            if tokens:
                yield tokens


    StopHeaderIteration = type('StopHeaderIteration', (StopIteration,), {})

    def iter_header(self, rawtokens):
        keyword_prefix = self.KEYWORD_PREFIX
        len_prefix = len(keyword_prefix)
        stop_token = keyword_prefix + self.END_OF_HEADER_NAME
        str_startswith = str.startswith

        for tokens in rawtokens:
            # print "header tokens:", tokens
            assert tokens
            token_iter = iter(tokens)
            for token in token_iter:
                if token == stop_token:
                    raise self.StopHeaderIteration                    
                if str_startswith(token, keyword_prefix):
                    keyword, args = token[len_prefix:], list(token_iter)
                    if len(args) != 1:
                        raise ValueError("expected at exactly one argument for keyword %s, got %s" % (repr(keyword), ''.join(repr(arg) for arg in args), ))
                    yield keyword, args[0]


    StopDataIteration = type('StopDataIteration', (StopIteration,), {})

    def iter_data(self, rawtokens):
        """
        Generator: for each line, yields a tuple of the decoded tokens
        """
        stop_token = self.KEYWORD_PREFIX + self.END_OF_DATA_NAME
        empty = self.EMPTY
        comment_prefix = self.comment_prefix
        escape_char = self.escape_char
        str_startswith = str.startswith

        import re
        decode_re = re.escape(escape_char) + r"(?:([0123456789abcdefABCDEF]{2})|('')|(.{,2}))"
        def replacer(match):
            hex_match, empty_match, bad_match = match.group(1,2,3)
            if hex_match is not None:
                assert empty_match is bad_match is None
                return chr(int(hex_match, 16))
            if empty_match is not None:
                assert hex_match is bad_match is None
                return empty

            assert bad_match is not None
            assert hex_match is empty_match is None
            raise ValueError("invalid escaping in %s" % (repr(match.string),))
        decode = _partial(re.compile(decode_re).sub, replacer)

        for tokens in rawtokens:
            assert tokens
            token0 = tokens[0]
            if str_startswith(token0, comment_prefix):
                continue
            if token0 == stop_token:
                raise self.StopDataIteration
            decoded_tokens = tuple(token if escape_char not in token else decode(token) for token in tokens)
            yield decoded_tokens

    
class TextdataReader(TextdataBase):
    """
    Reads the Textdata for an object from a stream, returning the unencoded tokens

    Gives access to the header information
    """

    def __init__(self, instream,
                 file_type=None,
                 file_version=None,
                 comment_prefix='#',
                 escape_char='%',
                 headerless=False):

        rawtokens = self.iter_rawtokens(instream)

        self.gets_header = not headerless

        # set these because they legitimately may not get overwritten
        self._file_type = self._file_version = None

        if self.gets_header:
            # parse the header

            next_keyword = self.iter_header(rawtokens).next

            def next_keyword_value(expected_name, expected_value=None):
                keyword, value = next_keyword()
                if keyword != expected_name:
                    raise ValueError("expected keyword %s, got %s with value %s" % (repr(self.KEYWORD_PREFIX+expected_name), repr(self.KEYWORD_PREFIX+keyword), repr(value),))
                if expected_value is not None and value != expected_value:
                    raise ValueError("expected keyword %s to have value %s, got %s" % (repr(self.KEYWORD_PREFIX+expected_name), repr(expected_value), repr(value),))
                return value

            try:
                dummy_version = next_keyword_value(self.VERSION_NAME, self.VERSION_VALUE)
                self.comment_prefix = next_keyword_value(self.COMMENT_PREFIX_NAME)
                self.escape_char = next_keyword_value(self.ESCAPE_CHAR_NAME)

                try:
                    self._file_type = next_keyword_value(self.FILE_TYPE_NAME)
                except self.StopHeaderIteration:
                    pass
                else:
                    self._file_version = next_keyword_value(self.FILE_VERSION_NAME)
                    # it should end here
                    try:
                        _ = next_keyword_value(self.END_OF_HEADER_NAME)
                    except self.StopHeaderIteration:
                        pass
                    else:
                        assert False, "unreachable"

            except self.StopHeaderIteration:
                raise ValueError("unexpected end of header while parsing header")
            except StopIteration:
                raise ValueError("unexpected end of stream while parsing header")

        else:
            if file_type is not None or file_version is not None:
                raise ValueError("expected None for file_type and file_version because headerless is %s, got %s and %s respectively" % (headerless, repr(file_type), repr(file_version),))
            self.comment_prefix = comment_prefix
            self.escape_char = escape_char
                
        self.check_configuration()

        if file_type is not None and self.file_type != file_type:
            raise ValueError("expected file_type %s, got %s" % (repr(file_type), repr(self.file_type),))

        if file_version is not None and self.file_version != file_version:
            raise ValueError("expected file_version %s, got %s" % (repr(file_version), repr(self.file_version),))

        self.data_iter = self.iter_data(rawtokens)
        
    def __iter__(self):
        return self.data_iter

class TextdataWriter(TextdataBase):
    """
    Writes the encoded Textdata tokens for an object to a stream

    Does the checking necessary to ensure that the contents written to
    the stream are a valid instance of a Textdata object
    """

    def __init__(self, outstream,
                 file_type=None,
                 file_version=None,
                 comment_prefix='#',
                 escape_char='%',
                 initial_newline=True,
                 headerless=False):

        # we set this False just before returning
        self.closed = True

        self._file_type = file_type
        self._file_version = file_version
        self.comment_prefix = comment_prefix
        self.escape_char = escape_char
        self.gets_header = not headerless

        self.check_configuration()

        self.writeit = outstream.write
        self.pred = None if initial_newline else self.NEWLINE
        self.wrote_header = False
        
        # figure out which characters are always encoded: start with
        # all characters, and then remove the graphical ones because
        # they don't need to be encoded, and then add in the special
        # ones which do need to be encoded, and finaly remove any
        # globally illegal ones

        work = set(chr(x) for x in xrange(256))
        work -= self.GRAPHICAL
        # note: a single-character string is its own iterable
        work |= frozenset(self.escape_char)
        # lazy: encode all characters that are part of comment prefix
        work |= frozenset(self.comment_prefix)
        work.add(self.EMPTY)
        work -= self.NOT_ENCODEABLE
        
        encode_char = self._encode_char
        encoder = self.encoder = dict((char, encode_char(char),) for char in work)

        # checks on the encodings
        assert self.EMPTY in encoder
        for not_encodeable in self.NOT_ENCODEABLE:
            assert not_encodeable not in encoder
        for encoded in encoder.itervalues():
            assert isinstance(encoded, str) and len(encoded) == 3, repr(encoded)

        # XXX do the speed and privacy optimizations of pulling the
        # function definitions into this scope and then binding the
        # public ones to self as static functions

        self.closed = False

    def write_header(self):
        if self.closed:
            raise ValueError("attempt to write to object after it was closed")
        if not self.gets_header:
            raise ValueError("invalid call to write_header(): perhaps this is a headerless writer or you've already (implicitly) written the header")

        def write_keyword_newline(*args):
            assert args
            args = list(args)
            args[0] = self.KEYWORD_PREFIX + args[0]
            self._write_graphicals(args)
            self._write_string(self.NEWLINE)

        self._ensure_whitespace(self.NEWLINE)

        write_keyword_newline(self.VERSION_NAME, self.VERSION_VALUE)
        write_keyword_newline(self.COMMENT_PREFIX_NAME, self.comment_prefix)
        write_keyword_newline(self.ESCAPE_CHAR_NAME, self.escape_char)
        if self._file_type is not None:
            write_keyword_newline(self.FILE_TYPE_NAME, self._file_type)
            write_keyword_newline(self.FILE_VERSION_NAME, self._file_version)
        write_keyword_newline(self.END_OF_HEADER_NAME)

        self.gets_header = False
        self.wrote_header = True

    def write_comment(self, comment=''):
        """
        Write the comment string, followed by a newline
        """
        _check_instance(str, comment)
        if self.closed:
            raise ValueError("attempt to write to object after it was closed")
        if frozenset(comment) - self.LEGAL_COMMENT:
            raise ValueError("invalid characters in comment %s" % (repr(comment),))

        # no encoding
        self._ensure_whitespace(self.NEWLINE)
        self._write_graphicals((self.comment_prefix,))
        self._write_graphicals(comment.split())
        self._write_string(self.NEWLINE)

    def write_newline(self):
        """
        Writes a newline
        """
        if self.closed:
            raise ValueError("attempt to write to object after it was closed")
        self._write_string(self.NEWLINE)

    def write_tokens(self, iterable):
        """
        If necessary, ensures that the header has been written;
        encodes each string token from iterable and appends each
        encoded token to the output stream with spacing
        """

        if isinstance(iterable, str):
            # forbid an error-prone case
            raise ValueError("expected an interable that is not itself a string, got %s: use write_token() instead" % (repr(iterable),))

        if self.closed:
            raise ValueError("attempt to write to object after it was closed")

        if self.gets_header:
            self.write_header()

        NOT_ENCODEABLE = self.NOT_ENCODEABLE
        end_of_data_token = self.KEYWORD_PREFIX + self.END_OF_DATA_NAME
        get = self.encoder.get
        encode_char = self._encode_char
        joinnosp = ''.join

        def encoder(iterable):
            for token in iterable:
                _check_instance(str, token)
                if frozenset(token) & NOT_ENCODEABLE:
                    raise ValueError("illegal characters in token %s" % (repr(token),))

                # slightly awkward dealing with fact that empty strings don't iterate...
                encoded = joinnosp(get(char, char) for char in token) if token else get(token)
                # we must have generated something
                assert encoded, repr(token)

                if encoded == end_of_data_token:
                    # forcibly encode the first character
                    encoded = encode_char(encoded[0]) + encoded[1:]
                    assert encoded != end_of_data_token

                yield encoded

        self._write_graphicals(encoder(iterable))

    # some sugar

    def write_token(self, token):
        """Ensures that the header has been written; encodes the
        string token to the output stream with spacing"""
        self.write_tokens((token,))

    def write_tokens_newline(self, iterable):
        """Does write_tokens(iterable) followed by write_newline()"""
        self.write_tokens(iterable)
        self.write_newline()

    def write_token_newline(self, token):
        """Does write_token(token) followed by write_newline()"""
        self.write_token(token)
        self.write_newline()

    def close(self):
        """Writes markers to note the end of this textdata object in
        the stream; futher attempts to write will raise an error;
        close() can be called multiple times"""

        if not self.closed:
            if self.gets_header:
                self.write_header()
            else:
                self._ensure_whitespace(self.NEWLINE)
            if self.wrote_header:
                self._write_graphicals((self.KEYWORD_PREFIX + self.END_OF_DATA_NAME,))
                self._write_string(self.NEWLINE)
            self.closed = True

    # internals

    def _encode_char(self, char):
        if char:
            return "%s%02x" % (self.escape_char, ord(char),)
        else:
            assert char == self.EMPTY
            return self.escape_char + self.ESCAPED_EMPTY
    
    def _write_graphicals(self, iterable):
        # asserts that tokens from iterable are graphical; writes them
        # with spaces to output stream
        GRAPHICAL = self.GRAPHICAL
        SPACE = self.SPACE
        write_space  = _partial(self._write_string, SPACE)
        write_string = self._write_string

        # XXX optimize by inlining functionality of _write_string

        self._ensure_whitespace(SPACE)
        for graphical_token in iterable:
            assert frozenset(graphical_token) <= GRAPHICAL
            write_string(graphical_token)
            write_space()

    def _write_string(self, string):
        # lowest level writer of a string; verifies that it consists
        # of one or more legal characters
        assert not self.closed
        assert isinstance(string, str) and string and not (frozenset(string) - self.LEGAL), repr(string)
        self.writeit(string)
        self.pred = string[-1]

    def _ensure_whitespace(self, char):
        assert char in self.WHITE
        if self.pred != char:
            self._write_string(char)
        assert self.pred == char

                 


def _logreftest():
    numtests = numpass = numfail = 0

    from cStringIO import StringIO

    def dowriter(tdwriter, explicit_header):
        tdwriter.write_comment('stuff before the header')
        tdwriter.write_newline()

        tdwriter.write_header() if explicit_header else tdwriter.write_tokens_newline('first writeln, maybe triggered a header'.split())
        tdwriter.write_newline()
        tdwriter.write_comment('stuff after the header')
        tdwriter.write_newline()

        tdwriter.write_comment('building a line')
        tdwriter.write_tokens(('foo', 'bar', 'baz',))
        # empties do nothing
        tdwriter.write_tokens(())
        # but empty string gets escaped
        tdwriter.write_token('')
        tdwriter.write_tokens(())
        tdwriter.write_token('foo')
        tdwriter.write_tokens(('foo', 'bar', 'baz', '', ''))
        tdwriter.write_newline()

        tdwriter.write_newline()
        tdwriter.write_comment('line at once')
        tdwriter.write_tokens_newline(('howdy', 'doody',))

        tdwriter.write_newline()
        tdwriter.write_comment('escaped stuff')
        tdwriter.write_tokens_newline(('new', 'york', 'new york', 'newline', 'new\nline', 'tabstop', 'tab\tstop',))
        # tdwriter.writeln('new', 'york', 'new york', 'newline', 'new\nline', 'tabstop', 'tab\tstop', '\x00')

        tdwriter.write_newline()
        tdwriter.write_comment('comment stuff: only graphical + space, no escaping, e.g.: # % ^ // etc.')
        tdwriter.write_tokens('content stuff: escaping for escape_char and any comment_prefix characters, e.g.: # % ^ // etc.'.split())

        tdwriter.write_newline()
        tdwriter.write_comment('the end of data marker can appear in a comment: __textdata__end_of_data')
        tdwriter.write_comment('but, it gets minimally escaped in data (including headerless data) to support cut-and-paste):')
        tdwriter.write_token_newline('__textdata__end_of_data')
        tdwriter.write_token_newline(tdwriter.KEYWORD_PREFIX + 'end_of_data')

        tdwriter.write_newline()
        tdwriter.write_comment('big long line')
        # example of generator usage
        tdwriter.write_tokens_newline("%04x" % (i,) for i in xrange(256))
        # tdwriter.writeln("%04x" % (i,) for i in xrange(1 << 9))

        # tdwriter.write_newline()
        # tdwriter.write_comment('check legal')
        # tdwriter.write_tokens(('foo', 'bar', 'x\x00',))

        tdwriter.write_newline()
        tdwriter.write_comment('maybe here comes the end of data marker')
        

    numtests += 1
    td1 = StringIO()

    tdwriter3 = TextdataWriter(td1, escape_char='^', comment_prefix='//', initial_newline=True, headerless=True)
    assert tdwriter3.file_type is tdwriter3.file_version is None
##     dowriter(tdwriter3, False)
##     tdwriter3.close()

    tdwriter1 = TextdataWriter(td1, escape_char='%', initial_newline=True)
    dowriter(tdwriter1, True)
    tdwriter1.close()

    tdwriter2 = TextdataWriter(td1, file_type='x', file_version='0.0', escape_char='^', comment_prefix='//')
    assert tdwriter2.file_type == 'x'
    assert tdwriter2.file_version == '0.0'
    dowriter(tdwriter2, False)
    tdwriter2.close()

    TextdataWriter(td1, file_type='OnyxGraph', file_version='0.0', escape_char='^', comment_prefix='//')
    TextdataWriter(td1, file_type='x', file_version='.')

    td1.seek(0)
    for line in td1:
        print line,

    numpass += 1
    assert numpass + numfail == numtests


    numtests += 1
    td1.seek(0)

    print
    print "tdreader1:"
    # tdreader1 = TextdataReader(td1, headerless=True)
    tdreader1 = TextdataReader(td1)
    for line in tdreader1:
        print line
        

    print
    print "tdreader3:"
    # tdreader3 = TextdataReader(td1, headerless=True)
    tdreader3 = TextdataReader(td1)
    for line in tdreader3:
        print line
        
#    tdreader1 = TextdataReader(td1)

    numpass += 1
    assert numpass + numfail == numtests


    print "numtests", numtests, " numpass", numpass, " numfail", numfail
    assert numpass + numfail == numtests
   


def _main(argv):
    import os.path
    prog = os.path.basename(argv[0])
    args = argv[1:]

    print prog, args

    usage = "%s --logreftest" % (prog,)
    
    if (len(args) > 0 and args[0] == '--logreftest') or prog == '':
        _logreftest()
        return

    print "usage:", usage
    print
    raise ValueError("invalid or missing args: %s" % (' '.join(map(repr, args)),))


if __name__ == '__main__':
    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    if len(argv) > 1:
        _main(argv)
