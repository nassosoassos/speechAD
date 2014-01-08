###########################################################################
#
# File:         tdutil.py
# Date:         17-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Low level utilities for dealing with textual data files
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
Some low-level utilities for working with textual data files.

This module is suitable for globbed import; all symbol names begin
with 'td' or 'Textdata'.

XXX Needs real testing.
    >>> True
    True
"""

from __future__ import with_statement as _with_statement

from sys import argv as _argv, stdin as _stdin, stdout as _stdout
from itertools import chain as _chain, izip as _izip, count as _count

# command-line stuff
# tdargv = _argv[:]
tdargs = _argv[1:]
tdwrite = _stdout.write

# common string joiners
tdjoinns = ''.join
tdjoinsp = ' '.join
tdjoinspnlsp = ' \n '.join


# exceptions

# the base_exception_spec is the name and doc for a base_exception that derives from Exception
base_exception_spec = ( 'TextdataException', 'Base exception for all Textdata exceptions', )

# the derived_exception_specs are names and docs for exceptions the derive from the base_exception
derived_exception_specs = (
    ( 'TextdataCommandlineError', 'Raised when command-line structure or preconditions are not met' ),
    ( 'TextdataSizeError', 'Raised when number of tokens is outside a specified range' ),
    ( 'TextdataParseFailure', 'Raised when text structure or preconditions lead to a parsing failure' ),
    ) #

# make the exceptions and inject them into the module's global namespace
def make_exceptions(namespace, base, deriveds):
    def make_exception(name, base, doc):
        exception = namespace[name] = type(name, (base,), {'__doc__':doc,})
        return exception
    name, doc = base
    baseexception = make_exception(name, Exception, doc)
    for name, doc in deriveds:
        make_exception(name, baseexception, doc)
make_exceptions(globals(), base_exception_spec, derived_exception_specs)
# cleanup
del make_exceptions, base_exception_spec, derived_exception_specs

def tdfileread(filename, flags='r'):
    """return contents of a file as a string"""
    with file(filename, flags) as infile:
        return infile.read()

def tdfileiter(filename, flags='r'):
    """return iterator for a file"""
    return iter(file(filename, flags))

# a subclass of list for the textdata tokens from a line of text; it
# has read-only atributes (name, line, text) to keep track of the
# source name (usually the filename), the line number, and the
# original line of text
#
# these are used for speed
_list_new = list.__new__
_list_init = list.__init__
class tdtokens(list):
    """a list that keeps track of a name and a line number"""
    __slots__ = ('name', 'line', 'text', '_name', '_line', '_text',)
    @property
    def name(self): return self._name
    @property
    def line(self): return self._line
    @property
    def text(self): return self._text
    def __new__(cls, name, line, text, args):
        return _list_new(cls, args)        
    def __init__(self, name, line, text, args):
        _list_init(self, args)
        self._name = name
        self._line = line
        self._text = text

_str_split = str.split
_unknownname = '<unknown>'
def _splitter(stringiterable, name=_unknownname):
    if name is _unknownname and hasattr(stringiterable, 'name'):
        name = stringiterable.name
    return iter(tdtokens(name, line, *text__tokens)
                for line, text__tokens in _izip(_count(1), iter((text, _str_split(text),)
                                                                  for text in stringiterable)) if text__tokens[1])
    
def tdnormalize(stringiterable, comment_prefix=None, name=_unknownname):
    """generator normalizes strings into textdata tokens, skipping blank lines and comments"""
    splitter = _splitter(stringiterable, name)
    if comment_prefix is None:
        return splitter
    def decommenter(next=splitter.next, comment_prefix=comment_prefix):
        startswith = str.startswith
        while 1:
            tokens = next()
            if not startswith(tokens[0], comment_prefix):
                yield tokens
    return decommenter()

_atleast = 'least'
_atmost = 'most'
def tdchecksizelimit(atlimit, atspec, tokensiterable):
    """atlimit is limit of legal zero-based index"""
    next = iter(tokensiterable).next
    assert 0 <= atlimit
    atlimit += 1
    assert atspec == _atleast or atspec == _atmost
    atleastNone = None if atspec == _atleast else False
    while 1:
        tokens = next()
        size = len(tokens)
        if (size < atlimit if atleastNone is None else size > atlimit):
            msg = "expected token count of at %s %d, got %d" % (atspec, atlimit, size,)
            if type(tokens) is tdtokens:
                # note: we strip the text here since tdtokens doesn't because it's used so rarely
                msg += ": %s(%d): %s" % (tokens.name, tokens.line, repr(tokens.text.strip()),)
            else:
                msg += " in %s" % (tokens,)
            raise TextdataSizeError(msg)
        yield tokens

def tdcheckatleast(atleast, tokensiterable):
    return tdchecksizelimit(atleast, _atleast, tokensiterable)

def tdcheckatmost(atmost, tokensiterable):
    return tdchecksizelimit(atmost, _atmost, tokensiterable)

def tddict(keycolumn, tokensiterable):
    assert 0 <= keycolumn
    return dict((tokens[keycolumn], tokens,) for tokens in tdcheckatleast(keycolumn, tokensiterable))
    

# multiple files iterators
def tdfilesiters(filenames, default=_stdin):
    """return tuple of iterators for the sequence of files, or the default"""
    fileiters = tuple(tdfileiter(arg) for arg in filenames) if filenames else (iter(default),)    
    return fileiters

# multi-file chain iterator
def tdfilesiterschain(filenames, default=_stdin):
    """return chained iterator for the sequence of files, or the default"""    
    return _chain(*tdfilesiters(filenames, default))

tdfilesiter = tdfilesiterschain

# multi-file izip iterator
def tdfilesitersizip(filenames, default=_stdin):
    """return iziped iterator for the sequence of files, or the default"""    
    return _izip(*tdfilesiters(filenames, default))

# XXX make the izip guy use a with that ensures that all file iterators got exhausted


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
