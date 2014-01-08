###########################################################################
#
# File:         site_init.py  (directory: ./site_scons)
# Date:         28-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Onyx site-specific SCons initialization checks
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
    Onyx site-specific SCons initialization checks.

    By virtue of its location relative to the top-level SConstruct file, this
    script gets run each time SCons is started.  It's where we do some basic
    checks on the configuration.

    >>> min_version
    (2, 5)

    >>> sys.version_info >= min_version
    True

    >>> project_py_dir
    'py'
    >>> os.path.split(onyx_package_mod)
    ('onyx', '__init__.py')
"""

min_version = 2, 5
import sys, os
if sys.version_info < min_version:
    raise ValueError("Onyx SCons configuration %r requires Python version of at least %s, got %s"
                      % (__name__, '.'.join(map(str, min_version)), '.'.join(map(str, sys.version_info[:3]))))

# Wow, here's a lot of work to help avoid headaches for people when they fiddle their PYTHONPATH
#
# We have to deal with three cases of this module being run:
# - by SCons as part of its startup
# - from stdin, e.g. from emacs ^C^C
# - by a Python for a doctest, spawned by SCons which fiddles PYTHONPATH

# for creating guess_onyx_path we are relying on SCons having done a chdir() to
# the top of the project, which, as of SCons 1.2.0, it does, even when called
# with -u from a subdirectory; or we're being run from an emacs buffer; or we're
# being run from a Python that SCons spawned as part of a doctest
project_py_dir = 'py'
# usual case
project_dir = os.getcwd()
if __file__ == '<stdin>':
    # when this script is run from an emacs buffer (via ^C^C) we move up one
    # directory to find the project
    project_dir, _ = os.path.split(project_dir)
    build_dir = '.'
else:
    # script is being run by SCons or a Python doing doctest, so project_dir is
    # ok and build_dir is up one directory above __file__
    build_dir, _ = os.path.split(__file__)
    build_dir, _ = os.path.split(build_dir)
guess_onyx_path = os.path.abspath(os.path.join(project_dir, build_dir, project_py_dir))

# make sure sys.path points into a project, e.g. via PYTHONPATH
try:
    import onyx
except ImportError:
    raise ImportError('No package named onyx: the absolute path to the project py directory '
                      'should be on your PYTHONPATH, e.g. perhaps %r' % (guess_onyx_path,))

# make sure that the onyx package that got loaded came from this project
actual_package_mod_abs = onyx.__file__
if actual_package_mod_abs.endswith('.pyc') or actual_package_mod_abs.endswith('.pyo'):
    actual_package_mod_abs = actual_package_mod_abs[:-1]
onyx_package_mod = os.path.join('onyx', '__init__.py')
assert actual_package_mod_abs.lower().endswith(onyx_package_mod), actual_package_mod_abs

# note: we require strict string equality, not just samefile equality because
# SCons does a chdir to the realpath
#if not os.path.samefile(actual_package_mod_abs, os.path.join(guess_onyx_path, onyx_package_mod)):
if actual_package_mod_abs.lower() != os.path.join(guess_onyx_path, onyx_package_mod).lower():
    raise ValueError(('Expected package onyx to come from the project in which SCons is being '
                      'run, %r, but it appears to come from a different project, %r: '
                      'perhaps your PYTHONPATH is pointing to the wrong project, or perhaps '
                      'your PYTHONPATH includes a symbolic link.  Try: export PYTHONPATH="%s"'
                      % (guess_onyx_path, actual_package_mod_abs[:-len(os.sep+onyx_package_mod)], guess_onyx_path)))

# eliminate duplicates in sys.path which SCons seems to put there...
new_path = list()
for path in sys.path:
    # n^2 work, n is usually small
    if path not in new_path:
        new_path.append(path)
sys.path = new_path


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
