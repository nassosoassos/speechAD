###########################################################################
#
# File:         ProjectTools.py
# Date:         16-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  SCons support for work
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
Top-level SCons initialization.

    >>> exists(None)
    True
"""

from __future__ import with_statement
import re
from functools import partial
from itertools import count
import distutils.sysconfig

from onyx.builtin import izipstrict, StrictError
from onyx.util import timestamp

# a simple diff tool
def diff_streams(env, target, source_ref, source_trial):

    # XXX: we need a proper normalization target to deal with all normalization
    # ; *this* function should be like diff

    assert target.name and source_ref.name and source_trial.name
    
    # mask out timestamps
    norm_timestamp = '###TIMESTAMP###'
    timestamp_subber = partial(timestamp.TIMESTAMP_RE.sub, norm_timestamp)

    # mask out pointer values, e.g. in 'ptr="0x00abcdef"'
    ptr_subber = partial(re.compile(r'\bptr\=\"0x[0-9a-fA-F]{1,8}\"').sub, r'###POINTER###')

    source_ref_iter = iter(source_ref)
    source_trial_iter = iter(source_trial)
    iters = count(1), source_ref_iter, source_trial_iter
    # assignment is necessary in case a file is empty
    lineno = 0
    try:
        for lineno, source_ref_line, source_trial_line in izipstrict(*iters):
            # XXX this would be useful if it did some string alignment work showing all diffs....

            source_ref_line = source_ref_line.rstrip('\n\r').replace('\\', '/')
            source_trial_line = source_trial_line.rstrip('\n\r').replace('\\', '/')

            source_ref_line = timestamp_subber(source_ref_line)
            source_trial_line = timestamp_subber(source_trial_line)
            
            source_ref_line = ptr_subber(source_ref_line)
            source_trial_line = ptr_subber(source_trial_line)

            if source_ref_line != source_trial_line:
                target.write("reference and trial files first differ on line %d\n" % (lineno,))
                target.write("%s(%d) : %s\n" % (source_ref.name, lineno, source_ref_line))
                target.write("%s(%d) : %s\n" % (source_trial.name, lineno, source_trial_line))
                return 1
    except StrictError, e:
        # we expect the count iterator to not have finished
        finished, unfinished, values = e.finished, e.unfinished, e.values
        assert unfinished and unfinished[0] == 0 and values[0] == lineno + 1
        if len(unfinished) > 1:
            assert len(finished) == 1
            target.write("%s ends at line %d : the other file continues\n" % (iters[finished[0]].name, lineno,))
            target.write("%s(%d) : %s\n" % (iters[unfinished[1]].name, lineno+1, values[1]))
            return 1
    else:
        assert False, "unreachable"

    return 0

from contextlib import nested
def diff(target, source, env):
    target, = target
    # note: ignore onyxtools_mod source
    source_ref, source_trial, _onyxtools_mod = source
    with nested(open(str(target), 'wt'),
                open(str(source_ref), 'rt'),
                open(str(source_trial), 'rt')) as files:
        return diff_streams(env, *files)

def generate(env):
    """
    Add construction variables and set up environment for Onyx SCons work
    """
    # set this to True for some logging while SCons is initializing

    debug = False

    import os   
    import SCons
    import onyx

    # set up some project-specific names in the SCons environment
    # print 'onyx.home:', repr(onyx.home)
    ONYX_BUILD = env['BUILD_TOP'] + '/' + onyx.platform
    env.SConsignFile(ONYX_BUILD + '/' + '.sconsign')

    ONYX_HOME = str(env.Dir(onyx.home))
    env['ONYX_HOME'] = ONYX_HOME
    env['ONYX_BUILD'] = env.Dir(ONYX_BUILD)
    env['ONYX_SOURCE_PYLIB'] = 'cpp/pylib/' + onyx.platform
    env['ONYX_BUILD_PYLIB'] = env.Dir(ONYX_BUILD + '/' + env['ONYX_SOURCE_PYLIB'])
    env['ONYX_BINPATH'] = onyx.bin_path
    env['ONYX_BUILD_BINPATH'] = onyx.home + '/' + ONYX_BUILD + '/' + 'bin' + '/' + onyx.platform
##     print 'ProjectTools: onyx_build:', ONYX_BUILD
##     print 'ProjectTools: onyx_source_pylib:', env['ONYX_SOURCE_PYLIB']
##     print 'ProjectTools: onyx_build_pylib:', env['ONYX_BUILD_PYLIB']
##     print 'ProjectTools: onyx_binpath:', env['ONYX_BINPATH']
##     print 'ProjectTools: onyx_build_binpath:', env['ONYX_BUILD_BINPATH']

##     # XXX obsolete
##     # tell SCons to unconditionally look in the platform-specific project directory first; this
##     # mirrors the behavior for Python's env found in py/onyx/__init__.py
##     env.PrependENVPath('PATH', env['ONYX_BINPATH'])

##     # SCons view of platform, less restrictive than our onyx.platform
##     PLATFORM = env['PLATFORM']
##     if PLATFORM == 'darwin':
##         env.PrependENVPath('PATH', '/opt/local/bin')

    shlibsuffix = distutils.sysconfig.get_config_var('SO')
    if shlibsuffix is None:
        raise ValueError("Don't know the proper extension for Python shared libraries on this platform")

    # _foo ==> _foo.so or _foo.pyd, depending on platform
    def make_pylib_name(basename):
        return basename + shlibsuffix

    # Add correct extension for shared libraries and a function for making names to env
    env.make_pylib_name = make_pylib_name

    # *this* generic tools file
    onyxtools_mod = env.onyxtools_mod = '#' + 'site_scons/site_tools/ProjectTools.py'

    def tools_dependency(target, source, env):
        if len(source) != 2:
            raise ValueError("expected exactly 2 sources for Diff builder, got %d" % (len(source),))
        return target, source + [onyxtools_mod]

    # the Diff builder uses the above diff() function
    # note: subscripting of SOURCES so as to avoid echoing the onyxtools_mod source
    env.Append(BUILDERS = {
        'Diff': env.Builder(
        action = SCons.Action.Action(diff, 'diff( ${SOURCES[:2]} > $TARGET )'),
        emitter = tools_dependency,
        )
        })

    # meta tools
    # managedfiles = list()
    # managedfiles.append('foo')
    # note: env.managed_sources gets built up while the nodes are getting defined as part of
    # SConscripts being run
    env.managed_sources = list()

    def ManagedFiles(filename):
        # print 'ManagedFiles():'

        def write_managedfiles_list(env, target, source):      
            # print 'write_managedfiles_list():', 'len(env.managed_sources)', len(env.managed_sources)
            with open(str(*target), 'wt') as outfile:
                for item in sorted(frozenset(env.managed_sources)):
                    outfile.write(str(item))
                    outfile.write('\n')

        from time import time
        # env.Command(filename, managed_sources, write_managedfiles_list)
        # XXX fix this so it actually does something
        #env.Command(filename, [env.managed_sources, env.Value(time())], write_managedfiles_list)
    env.ManagedFiles = ManagedFiles

    def LocalAllFilesInDir(dir_node, root):
        # print '>>>>', dir_node.abspath, dir_node.path, root
        assert dir_node.abspath == os.path.join(root, dir_node.path)
        for top, dirs, files in os.walk(dir_node.abspath):
            # print '>>>', root, dirs, files
            assert top.startswith(root)
            relative_top = top[len(root) + 1:]
            for f in files:
                # print '>>', os.path.join(relative_top, f)
                env.Local(os.path.join(relative_top, f))
    env.LocalAllFilesInDir = LocalAllFilesInDir

    def ProjectInstallProgram(source):
        # install sources into the project's bin directories
        env.Install(env['ONYX_BINPATH'], source)
        env.Install(env['ONYX_BUILD_BINPATH'], source)
    env.ProjectInstallProgram = ProjectInstallProgram


def exists(env):
    """
    Make sure that ProjectTools has the resources it needs
    """
    return True


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
