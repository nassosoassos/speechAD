###########################################################################
#
# File:         lint.py
# Date:         14-Aug-2009
# Author:       Hugh Secker-Walker
# Description:  Check copyright and license and other boilerplate
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2009 The Johns Hopkins University
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
Verify and/or generate various boilerplate for copyright and license.
"""
from __future__ import with_statement
import sys, os
import onyx.builtin

_test_header = """#############################################################################
##
## File:         lint.py
## Date:         14-Aug-2009
## Author:       Hugh Secker-Walker
## Description:  Check copyright and license and other boilerplate
##
## Copyright 2009, The Johns Hopkins University.  All rights reserved.
## Copyright 2009 Hugh Secker-Walker
##
###############################################################################

# first line
"""

_test_header2 = """#############################################################################
##
## File:         lint.py
## Date:         14-Aug-2009
## Author:       Hugh Secker-Walker
## Description:  Check copyright and license and other 
##               boilerplate
##
## Copyright 2009, The Johns Hopkins University.  All rights reserved.
## Copyright 2009, Onyx Tools Foundation.  All rights reserved.
##
###############################################################################

# first line
"""

class TextFileHeader(object):
    """
    Check and emit standard project header for text files.

    >>> import cStringIO
    >>> hdr = TextFileHeader(cStringIO.StringIO(_test_header))
    >>> hdr.directive, hdr.prefix
    (None, '')

    >>> for line in hdr: print line.rstrip()
    ###########################################################################
    ##
    ## File: lint.py
    ## Date: 14-Aug-2009
    ## Author: Hugh Secker-Walker
    ## Description: Check copyright and license and other boilerplate
    ##
    ## Copyright 2009, The Johns Hopkins University
    ## Copyright 2009 Hugh Secker-Walker
    ##
    ## Licensed under the Apache License, Version 2.0 (the "License").
    ## You may not use this file except in compliance with the License.
    ##
    ## You may obtain a copy of the License at
    ##   http://www.apache.org/licenses/LICENSE-2.0
    ##
    ## Unless required by applicable law or agreed to in writing, software
    ## distributed under the License is distributed on an "AS IS" BASIS,
    ## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    ## implied.  See the License for the specific language governing
    ## permissions and limitations under the License.
    ##
    ###########################################################################
    """

    line_length = 75
    directives = '#!', '..'
    license = (
        r'Licensed under the Apache License, Version 2.0 (the "License").',
        r'You may not use this file except in compliance with the License.',
        r'',
        r'You may obtain a copy of the License at',
        r'  http://www.apache.org/licenses/LICENSE-2.0',
        r'',
        r'Unless required by applicable law or agreed to in writing, software',
        r'distributed under the License is distributed on an "AS IS" BASIS,',
        r'WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or',
        r'implied.  See the License for the specific language governing',
        r'permissions and limitations under the License.',
        )
    def __init__(self, striter, strict=True):
        next_line = enumerate(iter(striter)).next
        try:
            lineno, line = next_line()

            # directive
            self.directive = None
            for directive in self.directives:
                if line.startswith(directive):
                    self.directive = line
                    lineno, line = next_line()
                    break

            # firstline
            firstline = line.strip()
            startset = set(firstline)
            if strict and len(startset) != 1: raise Exception
            startchar = set(startset).pop()
            def check_commentstr(string):
                if strict and set(string.strip()) != startset: raise Exception
            def check_field(candidate, reference):
                if candidate.lower() != reference.lower(): raise Exception
                
            def get_field(line, fieldname):
                comment, field, value = line.split(None, 2)
                check_commentstr(comment)
                check_field(field, fieldname)
                return value

            # prefix
            self.prefix = line[:line.find(firstline)]

            self.boundary = startchar * (self.line_length - len(self.prefix))

            lineno, line = next_line()

            # commentstr
            line = line.strip()
            if strict or line.startswith(startchar):
                check_commentstr(line)
                self.commentstr = line
            
                lineno, line = next_line()

            # file
            self.file = get_field(line, 'File:')
            
            lineno, line = next_line()
            # date
            self.date = get_field(line, 'Date:')
            
            lineno, line = next_line()
            # Author
            self.Author = get_field(line, 'Author:')
            
            lineno, line = next_line()
            # Description
            self.Description = get_field(line, 'Description:')

            # comment line
            lineno, line = next_line()
            if strict or line.startswith(startchar):
                check_commentstr(line)
                lineno, line = next_line()

            # Copyright
            copyrights = self.copyrights = list()
            try:
                while True:
                    copyright = get_field(line, 'Copyright')
                    lineno, line = next_line()
                    copyright = copyright.replace('.', '')
                    stops = 'all', 'rights', 'reserved'
                    parts = tuple(part for part in copyright.split() if part.lower() not in stops)
                    copyright = ' '.join(parts)
                    copyrights.append(copyright)
            except:
                pass

            # comment line
            check_commentstr(line)
            
            # blank line
            lineno, line = next_line()
            check_commentstr(line)

        except:
            raise ValueError("invalid header data on line %d: %s" % (lineno+1, line.strip()))
        
    def __iter__(self):
        prefix = self.prefix
        def lineify(*strs): return '%s%s\n' % (prefix, ' '.join(strs))
        commentstr = self.commentstr

        if self.directive is not None: yield self.directive + '\n'
        yield lineify(self.boundary)
        yield lineify(commentstr)
        yield lineify(commentstr, 'File:', self.file)
        yield lineify(commentstr, 'Date:', self.date)
        yield lineify(commentstr, 'Author:', self.Author)
        yield lineify(commentstr, 'Description:', self.Description)
        yield lineify(commentstr)
        for copyright in self.copyrights:
            yield lineify(commentstr, 'Copyright', copyright)
        yield lineify(commentstr)
        for license in self.license:
            yield lineify(commentstr, license)
        yield lineify(commentstr)
        yield lineify(self.boundary)


class TextFileHeader2(object):
    """
    Try to find header boilerplate info from the first few lines of a file.

    Check and emit standard project header for text files.

    >>> import cStringIO
    >>> hdr = TextFileHeader2(cStringIO.StringIO(_test_header2))
    >>> for line in hdr: print line.rstrip()
    ###########################################################################
    ##
    ## File:         lint.py
    ## Date:         14-Aug-2009
    ## Author:       Hugh Secker-Walker
    ## Description:  Check copyright and license and other boilerplate
    ##
    ## This file is part of Onyx   http://onyxtools.sourceforge.net
    ##
    ## Copyright 2009 The Johns Hopkins University
    ## Copyright 2009 Onyx Tools Foundation
    ##
    ## Licensed under the Apache License, Version 2.0 (the "License").
    ## You may not use this file except in compliance with the License.
    ## You may obtain a copy of the License at
    ##   http://www.apache.org/licenses/LICENSE-2.0
    ##
    ## Unless required by applicable law or agreed to in writing, software
    ## distributed under the License is distributed on an "AS IS" BASIS,
    ## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    ## implied.  See the License for the specific language governing
    ## permissions and limitations under the License.
    ##
    ###########################################################################
    """

    directives = '#!', '..'
    comments = frozenset('#/-=;:_')
    line_length = 75
    license = (
        r'Licensed under the Apache License, Version 2.0 (the "License").',
        r'You may not use this file except in compliance with the License.',
        r'You may obtain a copy of the License at',
        r'  http://www.apache.org/licenses/LICENSE-2.0',
        r'',
        r'Unless required by applicable law or agreed to in writing, software',
        r'distributed under the License is distributed on an "AS IS" BASIS,',
        r'WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or',
        r'implied.  See the License for the specific language governing',
        r'permissions and limitations under the License.',
        )

    @staticmethod
    def field_attr(field_name):
        return field_name.lower().replace(':', '')

    def __init__(self, striter):
        next_line = enumerate(iter(striter)).next

        fields = self.fields = onyx.builtin.attrdict()
        field_names = 'File:', 'Date:', 'Author:', 'Description:', 'Copyright'
        self.singleton_fields = tuple(field_name for field_name in field_names if field_name[-1] == ':')
        for field_name in field_names:
            fields[self.field_attr(field_name)] = list()
        fields.unknown = list()
        fields.comment_length = 1

        try_state = onyx.builtin.attrdict()
        try_state.last_field = None
        def try_fields(parts):
            lowered = parts[0].lower()
            value = ' '.join(parts[1:])
            if lowered.capitalize() in field_names:

                if lowered == 'copyright':
                    # normalize the copyright
                    # no periods
                    copyright = value.replace('.', '')
                    # remove 'all rights reserved'
                    stops = 'all', 'rights', 'reserved'
                    copyright_parts = list(part for part in copyright.split() if part.lower() not in stops)
                    # remove years commas that are not part of a years list,
                    # e.g. from 'Copyright 2007, The Johns Hopkins University'
                    for i in xrange(len(copyright_parts)-1):
                        pi = copyright_parts[i]
                        if pi in ('2007,','2008,','2009,') and copyright_parts[i+1][0] != '2':
                            copyright_parts[i] = pi[:-1]
                    if not (set(copyright_parts) <= set(('2007','2008','2009', '2007,','2008,', '-', 'The', 'Johns', 'Hopkins', 'University', 'Hugh', 'Secker-Walker', 'Onyx', 'Tools', 'Foundation'))):
                            raise ValueError('copyright: %s'%(' '.join(copyright_parts),))
                    copyright = ' '.join(copyright_parts)
                    value = copyright

                fields[self.field_attr(lowered)].append(value)
                try_state.last_field = lowered
            elif try_state.last_field == 'description:':
                # collapse multi-line descriptions
                fields[self.field_attr('description:')][0] += ' ' + parts[0] + ' ' + value
            else:
                fields.unknown.append(value)
                try_state.last_field = None
                
        try:
            lineno, line = next_line()

            # directive, if any
            fields.directive = None
            for directive in self.directives:
                if line.startswith(directive):
                    fields.directive = line.rstrip()
                    #print 'directive', 
                    lineno, line = next_line()
                    break

            # firstline
            firstline = line.strip()
            # indent
            fields.indent = line[:line.find(firstline)]

            startset = set(firstline)
            if len(startset) == 1:
                # standard first line
                startchar1 = set(startset).pop() 
                #print 'startchar1', startchar1
                lineno, line = next_line()
            else:
                startchar1 = None

            while True:
                if lineno >= 15: raise ValueError("headerless")

                line2 = line.strip()
                lineset = set(line2)
                if len(lineset) == 1:
                    fields.startchar = set(lineset).pop() 
                    if len(line2) > 20:
                        # standard final line
                        break
                    else:
                        lineno, line = next_line()
                        continue

                parts = line.split()
                if not parts:
                    lineno, line = next_line()
                    continue
                if set(parts[0]) < self.comments:
                    if len(parts) > 1 and len(parts[0]) > fields.comment_length:
                        # e.g. for C style '//' comments
                        fields.comment_length = len(parts[0])
                    del parts[0]
                if parts:
                    try_fields(parts)
                    lineno, line = next_line()

            if len(fields.unknown) > 0: raise ValueError("unknown field %r" % (fields.unknown[0],))
            #print fields
            if min(len(fields[self.field_attr(x)]) for x in field_names) == 0: raise ValueError("missing fields")
            if max(len(fields[self.field_attr(x)]) for x in field_names if x in self.singleton_fields) > 1 : raise ValueError("multivalued field")
            fields.file_header_lines = lineno + 1

        except:
            raise
            #raise ValueError("invalid header data on line %d: %s" % (lineno+1, line.strip()))
            
    def __iter__(self):
        fields = self.fields
        indent = fields.indent
        def lineify(*strs): return '%s%s\n' % (indent, ' '.join(strs))
        commentchar = fields.startchar
        boundary = commentchar * (self.line_length - len(indent))
        commentstr = commentchar * fields.comment_length

        max_field_name_len = max(len(field_name) for field_name in self.singleton_fields)
        def format_singleton(field_name):
            field_name_len = len(field_name)
            assert max_field_name_len >= field_name_len
            padding = max_field_name_len - field_name_len
            value, = fields[self.field_attr(field_name)]
            value = ' '.join(value.split())
            return '%s%s  %s' %(field_name, ' ' * padding, value)

        if fields.directive is not None:
            yield fields.directive + '\n'
        yield lineify(boundary)
        yield lineify(commentstr)
        for field_name in self.singleton_fields:
            yield lineify(commentstr, format_singleton(field_name))
        yield lineify(commentstr)
        yield lineify(commentstr, r'This file is part of Onyx   http://onyxtools.sourceforge.net')
        #yield lineify(commentstr, r'  http://onyxtools.sourceforge.net')
        yield lineify(commentstr)
        for copyright in fields.copyright:
            yield lineify(commentstr, 'Copyright', copyright)
        yield lineify(commentstr)
        for license in self.license:
            yield lineify(commentstr, license)
        yield lineify(commentstr)
        yield lineify(boundary)

def fixgood(filename, header, dry_run=False, suffix=''):
    outfilename = filename + suffix

    filemode = os.stat(filename).st_mode

    skip =  header.fields.file_header_lines
    with open(filename, 'rb') as infile:
        for i in xrange(skip):
            infile.next()
        file_contents = tuple(infile)

    file_lines = list()
    for line in header:
        file_lines.append(line)
    if dry_run:
        try:
            for i in xrange(skip):
                file_lines.append(file_contents[i])
        except IndexError:
            pass
    else:
        file_lines.extend(file_contents)
        
    if dry_run:
        for line in file_lines:
            print line,
        print
    else:
        with open(outfilename, 'wb') as outfile:
            for line in file_lines:
                outfile.write(line)
        os.chmod(outfilename, filemode)
        

def main_old():
    args = sys.argv[1:]
    dry_run = False
    while '--dry-run' in args:
        dry_run = True
        args.remove('--dry-run')
        
    if args and args[0] == '--good':
        mode = 'good'
        del args[0]
    elif args and args[0] == '--filename':
        mode = 'filename'
        del args[0]
    elif args and args[0] == '--fixgood':
        mode = 'fixgood'
        del args[0]
    else:
        mode = 'bad'
        
    if args:
        print 'headers = ('
        for filename in args:
            try:
                with open(filename, 'rb') as infile:
                    header = TextFileHeader2(infile)
            except Exception, e:
                #print filename
                msg = str(e) if str(e) else type(e).__name__
                print ' ', '#', repr(filename), ',', ' ', '#', msg
                pass
            else:
                if mode == 'filename':
                    header_filename = header.fields.file[0].split()[0]
                    if header_filename != os.path.split(filename)[-1]:
                        print ' ', repr(filename), ',', ' ', '#', 'filename:', header_filename
                elif mode == 'good':
                    print ' ', (filename, dict(header.fields)), ','
                elif mode == 'fixgood':
                    print ' ', (filename, dict(header.fields)), ','
                    fixgood(filename, header, dry_run=dry_run, suffix='')
                pass
        print ')'


class Lint(object):
    directives = '#!', '..'
    comments = frozenset('#/-=')
    def __init__(self, string_iterable):

        self._is_header_valid = False
        self._status = 'early failure'
        header = self.header = onyx.builtin.attrdict()
        header.num_lines = 0
        def next_line():
            next = iter(string_iterable).next
            while True:
                line = next()
                header.num_lines += 1
                yield line
        next_line = next_line().next

        try:
            line = next_line()

            # directive, if any
            header.directive = None
            for directive in self.directives:
                if line.startswith(directive):
                    header.directive = line.rstrip()
                    line = next_line()
                    break

            # firstline
            firstline = line.strip()
            
            comment_char = set(firstline)
            if len(comment_char) != 1:
                raise ValueError('multiple comment chars in first line: %r' % (firstline,))
            header.comment_char = set(comment_char).pop()
            # indent
            header.indent = line[:line.find(firstline)]



            while True:
                line = next_line()

        except StopIteration:
            self._status = 'EOF'
        except ValueError, e:
            self._status = str(e)

    @property
    def is_header_valid(self):
        return self._is_header_valid
    
    @property
    def status(self):
        return self._status

def main(args):
    for filename in args:
        with open(filename, 'rb') as infile:
            lint = Lint(infile)

##         if lint.header.directive is not None:
##             print filename, repr(lint.status), repr(lint.header.directive)
##         continue
        if not lint.is_header_valid:
            print filename, ' ', lint.status

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    main(sys.argv[1:])
