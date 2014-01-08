###########################################################################
#
# File:         gridgo.py (directory: ./sandbox/grid)
# Date:         4-Feb-2009
# Author:       Hugh Secker-Walker
# Description:  Support for function-like access to the compute grid
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
    >>> module_dir, module_name = os.path.split(__file__)

    >>> res = grid_do('onyx.grid.griddy', 'my_func', ((1,2,3), ('a', 'b', 'c')), dir=module_dir, fake_grid=True) #doctest: +ELLIPSIS
    pickle_files: ...grid_do.0.pkl ...grid_do.1.pkl
    run_function: my_func ( 1, 2, 3 )
    run_function: my_func ( 'a', 'b', 'c' )
    >>> res
    [(1, 2, 3), ('a', 'b', 'c')]
"""
from __future__ import with_statement
import os, sys
import cPickle
from subprocess import Popen, PIPE, STDOUT


PICKLE_TYPE = 'grid_job'
PICKLE_VERSION = 0

FUNCTION_CALL = 'function_call'

def grid_do(package, function, args_iterable, dir='', fake_grid=False):
    vars = dict()
    x = __import__(package, globals(), vars, [function], 0)
    assert not vars
    y = getattr(x, function)

    argsen = tuple(tuple(arg for arg in args) for args in args_iterable)

    base_name = 'grid_do'
    suffix = 'pkl'
    num_digits = len('%d' % (len(argsen),))
    names = list()
    for index, args in enumerate(argsen):
        name = os.path.join(dir, '%s.%0*d.%s' % (base_name, num_digits, index, suffix))
        with open(name, 'wb') as pickle_outfile:
            def dump(obj): cPickle.dump(obj, pickle_outfile)
            dump(PICKLE_TYPE)
            dump(PICKLE_VERSION)
            dump(FUNCTION_CALL)
            dump(package)
            dump(function)
            dump(args)
        names.append(name)
    print 'pickle_files:', ' '.join(names)

    if fake_grid:
        resen = list()
        for name in names:
            resen.append(runjob((name,)))
        return resen
    else:
        pass

def run_function(load):
    package = load()
    function = load()
    args = load()

    vars = dict()
    mod = __import__(package, globals(), vars, [function], 0)
    assert not vars
    func = getattr(mod, function)
    
    print 'run_function:', func.__name__, '(', ', '.join(repr(x) for x in args), ')'
    
    return func(*args)
    
def runjob(args):
    if len(args) != 1:
        raise ValueError("expected 1 argument, got %d" % (len(args),))
    arg, = args
    with open(arg, 'rb') as pickle_infile:
        def load(): return cPickle.load(pickle_infile)
        pickle_type = load()
        pickle_version = load()
        if pickle_type != PICKLE_TYPE or pickle_version != PICKLE_VERSION:
            raise ValueError("expected pickled header of (%r, %r), got (%r, %r)" % (PICKLE_TYPE, PICKLE_VERSION, pickle_type, pickle_version))
        job_type = load()
        if job_type == FUNCTION_CALL:
            result = run_function(load)
        else:
            raise ValueError("unexpected job_type: %r" % (job_type,))

        return result

def _logreftest():
    # run one of the pickled jobs that doctest made; do this in a different
    # Python

    # note: can't put this in the doctest, or else an infinite process loop
    # occurs!

    module_dir, module_name = os.path.split(__file__)
    args = (
        sys.executable,
        '-m',
        'onyx.grid.gridgo',
        os.path.join(module_dir, 'grid_do.1.pkl')
        )
    proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=False)
    stdout, stderr = proc.communicate()
    assert proc.returncode is not None
    if proc.returncode != 0:
        raise ValueError("command failed: %r : %s" % (' '.join(args), stderr.strip()))
    print stdout


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    import sys
    args = sys.argv[1:]
    if args:
        if '--logreftest' in args:
            _logreftest()
        else:
            res = runjob(args)
            print res
