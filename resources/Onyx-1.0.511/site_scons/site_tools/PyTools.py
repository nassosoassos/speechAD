###########################################################################
#
# File:         PyTools.py
# Date:         2-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  SCons tools to support project work on Python files
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

r"""
SCons tools for compiling and testing Python code.

>>> exists(None)
True

Some quick tests of parse_doctest_message, a onyx tool we rely on:
>>> parse_doctest_message('foo bar') is None
True

A parseable message, successful tests
>>> parse = parse_doctest_message('doctestmod : __main__ : total 1  ok 1  bad 0 : pass \n')
>>> parse
('doctestmod', '__main__', 1, 1, 0, 'pass')
>>> tag, module_name, total, ok, bad, passfail = parse
>>> tag == 'doctestmod', total >= 1, ok >= 1, bad == 0
(True, True, True, True)

A parseable message, one test failed
>>> parse_doctest_message("doctestmod : __main__ : total 2  ok 1  bad 1 : *** fail *** : file '__main__' \n")
('doctestmod', '__main__', 2, 1, 1, "*** fail *** : file '__main__'")
"""

from __future__ import with_statement

import os, sys

# note: the use of parse_doctest_message is a very rare instance of
# the SCons tools relying on something from the onyx package and thus
# on PYTHONPATH...
#
# XXX it would be very nice to not have this dependency, e.g. by doing
# explicit imports inside generate() via ONYX_HOME
from onyx import parse_doctest_message, DOCTESTMODTAG, home as onyx_home


# removing this restriction would require work when compiling PYC and PYO files
debug_only = ("SCons PyTools.py only works in a non-optimized Python runtime "
              "(i.e. __debug__ == True).  Remove the '-O' argument to Python.")

def generate(env):
   """
   Add builders and construction variables for building and testing
   Python code.
   """

   # XXX we need to establish conventions for path names, both
   # absolute and relative to the top of the project; and relatedly,
   # establish conventions for use of strings versus File and Dir
   # nodes....

   # one difficulty here is that SCons is not transparent (to me anyway -HSW)
   # about the paths associated with a node; I've seen where str(node) gives
   # different values depending on whether the environment is for a variant
   # build or at the top-level; I haven't (yet)seen such variability in
   # node.path, node.abspath

   # note: we are relying on ONYX_HOME and ONYX_BUILD_PYLIB having
   # been set already; see ProjectTools.py

   if not __debug__:
      # removing this restriction would require work when compiling PYC and PYO files
      raise AssertionError(debug_only)
   
##    ONYX_HOME = env['ONYX_HOME']
##    print repr(str(ONYX_HOME)), repr(onyx_home)
   ONYX_BUILD_PYLIB = str(env['ONYX_BUILD_PYLIB'])

   import SCons
   from os import path

   # note: we set up dependencies for most built files on some of the low-level
   # tools in the project; this means that changes to these tools result in
   # widespread rebuilding, as they should

   # project directory to put on Python path
   py_dir = 'py'

##    # XXX Should this be the source files or the built files?  Until we
##    # get the pyc dependency issue fixed it has to be the source files.
##    #ONYX_PYTHONPATH = path.abspath(path.join(ONYX_HOME, py_dir))
##    #print 'ONYX_HOME:', repr(ONYX_HOME)
##    #print 'ONYX_PYTHONPATH:', repr(ONYX_PYTHONPATH)
##    #print 'ONYX_BUILD:', repr(str(env['ONYX_BUILD']))

   # XXX HSW doesn't understand how things could work from a release if you
   # haven't done bin/scons...

   # set up PYTHONPATH in the environment for Pythons that we spawn; require
   # that all packages are loaded from the build tree
   ONYX_PYTHONPATH = path.abspath(path.join(str(env['ONYX_BUILD']), py_dir))
   env.AppendENVPath('PYTHONPATH', ONYX_PYTHONPATH)
   # Here we prepend the pylib build directory to the PYTHONPATH so that spawned
   # Pythons get shared objects from there
   env.PrependENVPath('PYTHONPATH', path.abspath(ONYX_BUILD_PYLIB))

   # some important files upon which our Python usage depends

   # set by ProjectTools.py
   onyxtools_mod = env.onyxtools_mod

   # this file
   pytools_mod = '#' + 'site_scons/site_tools/PyTools.py'

   # the onyx package module
   onyxpackage_mod = '#' + 'py/onyx/__init__.py'

   # the doctesting module
   doctestmod_mod = '#' + 'py/onyx/doctestmod.py'

   # XXX fewer files needed explicitly now that we have a scanner;
   # may not need onyxpackage_mod

   # XXX it would be nice to be able to have the dependencies be on the PYCs for
   # the tool_files, then we wouldn't rebuild if, e.g. only commentary was
   # changed, but the expedient of adding 'c' to the file names didn't work;
   # another idea that I haven't tried yet is for us to keep a version of the
   # PYC around with a masked-out timestamp and use *that* to decide if the PYCs
   # should be updated;

   tool_files = env.Flatten(list(env.File(file) for file in (onyxtools_mod, pytools_mod, onyxpackage_mod, doctestmod_mod)))   
   tool_files += env.Flatten(list(env.File(os.path.join('#', str(env['ONYX_BUILD']), str(file) + 'c')) for file in tool_files))
   #print 'tool_files:', type(tool_files), tuple(str(x) for x in tool_files)

   def add_managed_source(node, env, path):
      # print 'add_managed_source():', str(node)
      env.managed_sources.append(str(node))

   # a scanner to find dependencies for Python files, based on the modulefinder
   # module in Python
   from modulefinder import ModuleFinder
   onyx_home_l = onyx_home.lower() + os.sep
   onyx_build = str(env['ONYX_BUILD'])

   def python_import_scanner(node, env, path):
      # this is the scanner we use to decide if the doctests for a PYC should be
      # re-run; it's handed a PYC and other filenames, usually in the build
      # directory; but, because SCons doesn't guarantee that a target has been
      # built prior to calling the scanner for that target (see logging from
      # node_exists_check() and the SCons man-page regarding scanners) we have
      # to be clever; in this case, clever means figuring out the PY file in the
      # source that corresponds to the target PYC, and then using modulefinder
      # to get "dependencies"; these of course are in the source tree, so we
      # have to put the variant-build prefix back on them....  so, this scanner
      # knows way too much about what's going on with SCons

      debug_me = False
      add_managed_source(node, env, path)

      if debug_me: print 'python_import_scanner:', 'path:', tuple(str(x) for x in path)
      if debug_me: print 'python_import_scanner:', 'node:', repr(str(node)), repr(node.path), repr(node.abspath)

      if not (node.path.endswith('.pyc') or node.path.endswith('.py')):
         # we allow non python files, but we don't scan them, but we do list
         # them as dependencies; this causes them to get copied to the build
         # tree early enough for tests; see DocTest builder which allows
         # multiple sources
         assert not node.path.endswith('.pyo')
         if debug_me: print '+', repr(node.path)
         # These files may have been prefixed with '#' in the SConscript file,
         # in which case they won't have the build path prefix on them at this
         # point.  Or the SConscript may have used a relative name without a
         # '#', in which case they will have the prefix, so we have to check here.
         prefix = '#' if node.path.startswith(onyx_build) else os.path.join('#', onyx_build)
         dependency = os.path.join(prefix, node.path)
         if debug_me: print ' ', repr(dependency)
         return [dependency]

      # a nasty hack: DocTest uses this scanner, so it's called with PYC files
      # in the build tree *that may not exist yet* (as of SCons 1.2.0), so we
      # strip of the variant build portion of the name
      if node.path.startswith(onyx_build):
         script_filename = onyx_home + os.sep + node.path[len(onyx_build + os.sep):]
      else:
         script_filename = node.abspath

      # a bit of a hack: DocTest uses this scanner, so it's called with PYC
      # files, so we fiddle filenames so that run_script, which needs PY files
      # (the way we're using it anyway) works
      if script_filename.endswith('.pyc'):
         script_filename = script_filename[:-1]
      assert script_filename.endswith('.py')

      if debug_me: print 'script_filename:', script_filename
      finder = ModuleFinder(list(path))
      finder.run_script(script_filename)
      # argggg, ModuleFinder doesn't seem to implement a documented
      # programatic access to the dependencies, so, after looking at
      # the implementation of ModuleFinder.report() from Python 2.5.1,
      # we do the following to iterate through what it's learned
      dependencies = set()
      for value in finder.modules.values():
         filename = value.__file__
         if filename is not None:
            if debug_me: print '-', repr(filename)
            if filename.lower().startswith(onyx_home_l):
               filename = filename[len(onyx_home_l):]
               if debug_me: print ' ', repr(filename)
               assert filename.lower().endswith('.py')
               # returning filenames in the build tree causes the dependencies
               # to get built; as of SCons 1.2.0, alternatives, like returning
               # the filenames in the source tree, do not work; this feels like
               # a very grey area of SCons, that is, running the scanner on
               # targets of other builders rather than on sources, and returning
               # '#' prefixed filenames
               dependency = os.path.join('#', onyx_build, filename)
               #dependency = os.path.join('#', filename)               
               if debug_me: print ' ', repr(dependency)
               dependencies.add(dependency)
               # also report about PYCs so that dependencies get built and installed
               dependency += 'c'
               if debug_me: print ' ', repr(dependency)
               dependencies.add(dependency)

      return sorted(dependencies)

   def scanner_path_func(env, src_dir_node, target_nodes, source_nodes, arg=None):
      # return tuple of directories that should be on the (python_)path that we
      # hand to ModuleFinder; the first of these is debatable, it allows modules
      # to import other modules from their package without any prefix....
      return os.path.join(onyx_home, str(src_dir_node.srcnode())), os.path.join(onyx_home, py_dir)
      #return os.path.join(onyx_home, str(src_dir_node)), ONYX_PYTHONPATH
      #return str(src_dir_node), ONYX_PYTHONPATH
      #return ONYX_PYTHONPATH,
      #return os.path.join(onyx_home, str(src_dir_node.srcnode())), ONYX_PYTHONPATH

   def node_exists_check(node, env):
      exists = node.exists()
      # as of SCons 1.2.0 it can happen that we're asked to scan a non-existent target....
      if not exists: print 'PyTools.py: node_exists_check: not node.exists()', repr(str(node))
      # we return True because the scanner doesn't use the PYC-file, but instead
      # figures out and uses the PY-file
      return True

   python_scanner = env.Scanner(function=python_import_scanner,
                                path_function=scanner_path_func,
                                node_class=SCons.Node.FS.File,
                                node_factory=env.File,
                                scan_check=node_exists_check)
   
   # make sure the Python that's running SCons will be the Python that is used
   # for our spawned Python work
   from sys import executable as python_exec
   env['ONYX_PYTHON_EXEC'] = python_exec
   python_exec += ' '

   # Action function for compiling PYC files, these are done in typical SCons
   # fashion one at a time using the Python that's running SCons
   def pyc_compile_function(target, source, env):
      target, = target
      source, = source
      import py_compile
      py_compile.compile(source.path, target.path)
      #py_compile.compile(str(source), str(target))

##    #py_compile_module = '.'.join(py_compile_names)
##    #pyc_compile_command = python_exec + '-S -m ' + py_compile_module + ' $SOURCE '
##    # the command-line string for PYO compilation
##    #pyo_compile_command = python_exec + '-S -O -m ' + py_compile_module + ' $SOURCE '
##    pyo_compile_command = python_exec + '-S -O -m py_compile $SOURCE '   
##    pyo_compile_command = python_exec + '-S -O -m py_compile $SOURCES '   

   # list of files needing PYO compilation; this is a delayed, batched operation
   pyo_source_files = env['PYO_SOURCE_FILES'] = list()

##    def pyo_compile_function(target, source, env):
##       import os
##       sources = ' '.join(pyo_source_files)
##       print 'sources:'
##       print sources
##       os.system(env['ONYX_PYTHON_EXEC'] + ' -S -O -m py_compile ' + sources)
##    pyo_compile_action = SCons.Action.Action(pyo_compile_function)
##    pyo_compile_action = SCons.Action.Action(env['ONYX_PYTHON_EXEC'] + " -S -O -m py_compile ${' '.join(env.pyo_source_files)}")
##    #pyo_compile = env.Command('foo.pyo.compile', env.Value(None), pyo_compile_action)
##    def hello():
##       print 'hello:'
##       ret = None
##       print ret
##       return ret
##    #pyo_compile = env.Command(env.Value('pyo_compile'), env.Value(None), hello())
##    #pyo_compile = env.Command(env.Value('pyo_compile'), env.Value(None),
##    #pyo_compile = env.Command('pyo_compile', env.Value(pyo_source_files),

   # this guy is a hack; because we don't actually create the file 'pyo_compile'
   # the command always runs, and it uses the delayed evaluation feature of
   # python code in ${...} to get the list of files that are actually out of
   # date; what would be better is an action that is only run if
   # PYO_SOURCE_FILES is not empty; need to learn more SCons
   pyo_compile = env.Command('pyo_compile', [],
                             env['ONYX_PYTHON_EXEC'] + " -S -O -m py_compile ${' '.join(PYO_SOURCE_FILES)}")
   def pyo_register_function(target, source, env):
      target, = target
      source, = source
      # an enabler of the hack: we use the env.Pyo builder's action to grow the
      # list of pyo_source_files with PYO filenames for out-of-date PYOs
      pyo_source_files.append(source.path)      

   # the .pyc and .pyo builders
   env.Append(BUILDERS = {
      'Pyc': env.Builder(
         src_suffix = 'py',
         single_source = True,

         suffix = 'pyc',
         ensure_suffix = True,

         # this compiles the PY file using the Python that's running SCons
         action = SCons.Action.Action(pyc_compile_function, 'py_compile_pyc( $SOURCE )'),
      ),
      'Pyo': env.Builder(
         src_suffix = 'py',
         single_source = True,

         suffix = 'pyo',
         ensure_suffix = True,

         # this action just collects the names of the PYOs
         action = SCons.Action.Action(pyo_register_function, 'register py_compile_pyo $SOURCE'),
      ),
      })

   # the doctest runner and verifier
   env.Append(BUILDERS = {
      'DocTest': env.Builder(
      src_suffix = 'pyc',
      # allow multiple files as sources, this allows the dependency nodes to be
      # in the variant-build tree
      source_scanner = python_scanner,

      suffix = 'log-doctest',
      ensure_suffix = True,

      action = SCons.Action.Action(python_exec + '$SOURCE > $TARGET'),
      ),
      })
   
   # the logreftest runner and verifier
   env.Append(BUILDERS = {
      'LogrefTest': env.Builder(
      src_suffix = 'pyc',
      single_source = True,
      # gack! it's time consuming to scan again for these tests....  hmmm, it's
      # possible this could be avoided by careful use of env.Depends in PyFile()
      source_scanner = python_scanner,

      suffix = 'log-logreftest',
      ensure_suffix = True,

      action = SCons.Action.Action(python_exec + '$SOURCE --logreftest > $TARGET'),
      ),
      })

   # Global Accumulators

   # note: managedmodules and files_for_doc_build get built up while the nodes
   # are getting defined as part of SConscripts being run
   managedmodules = list()
   files_for_doc_build = dict()
   nodes_for_doc_build_dependency = list()
   
   # note: doctestsummary gets built up as part of the actions getting run
   doctestsummary = list()

   
   # verify doctest logs, accumulate stats

   def verify_log_doctest(target, source, env):
      target, = target
      source, = source
      sourcename = str(source)
      with open(sourcename, 'rt') as infile:
         lines = infile.readlines()
      
      # error code 3 if there's more than one line
      if len(lines) != 1:
         print >> sys.stderr, 'error: ',
         print "expected 1 line, got %d, in file %r" % (len(lines), sourcename,)
         return 3
      line, = lines

      # error code 2 if the line is poorly formatted
      parts = parse_doctest_message(line)
      if parts is None:
         print >> sys.stderr, 'error: ',
         print "unexpected: parts is None in file %r" % (sourcename,)
         return 2
      tag, module_name, total, ok, bad, passfail = parts
      if tag != DOCTESTMODTAG:
         print >> sys.stderr, 'error: ',
         print "expected tag of %r, got %r, in file %r" % (DOCTESTMODTAG, tag, sourcename,)
         return 2

      doctestsummary.append((total, ok, bad))

      # error code 1 if there were no tests or if any tests failed
      if total == 0 or bad != 0:
         print >> sys.stderr, 'error: ',
         print "unexpected values of total %d or bad %d in file %r" % (total, bad, sourcename,)
         return 1

      # error code 4 if we can't create the intentionally-empty target file
      try:
         open(str(target), 'wt').close()
      except IOError:
         print >> sys.stderr, 'error: ',
         print "unexpected IOError while opening or closing file %r" % (str(target),)
         return 4
         
      # error code 0 if all is OK
      return 0

   # the doctest runner and verifier
   env.Append(BUILDERS = {
      'DocTestVerify': env.Builder(
      src_suffix = 'log-doctest',
      single_source = True,

      suffix = 'log-doctestverify',
      ensure_suffix = True,

      action = SCons.Action.Action(verify_log_doctest, 'verify_log_doctest( $SOURCE > $TARGET )'),
      ),
      })

   def log_logreftest_emitter(target, source, env):
      # yuk, but we want to depend on onyxtools_mod and the diff
      # builder in ProjectTools.py needs to see a third source (that it then
      # ignores)
      return target, source + [onyxtools_mod]

   from ProjectTools import diff
   def rev_diff(target, source, env):
      s1 = source[:]
      # reverse the two args to match ProjectTools.diff's semantics, the third
      # argument remains
      s1[0], s1[1] = s1[1], s1[0]
      return diff(target, s1, env)

   # the logreftest runner and verifier
   # note: subscripting of SOURCES so as to avoid echoing the onyxtools_mod source
   env.Append(BUILDERS = {
      'LogrefTestVerify': env.Builder(
      src_suffix = 'log-logreftest',
      single_source = False,

      suffix = 'log-logreftestverify',
      ensure_suffix = True,

      emitter = log_logreftest_emitter,
      action = SCons.Action.Action(rev_diff, 'diff( ${SOURCES[1]}  ${SOURCES[0]} > $TARGET )'),
      ),
      })


   def DocBuildThisPackage(pkg_title, categories=None):
      """
      Initialize doc-building for a package.  The *categories* argument, if not
      None, should be a list of pairs (cat_name, cat_doc) where both elements
      are strings.  Some cat_name must then be used for each PyFile declaration
      for this package.

      Package-level SConscript files should call this function *before* calling
      PyFile if they want to include their PyFiles in the automatically generated
      documentation.  Control is also available for individual PyFile calls using
      the "no_doc" argument.
      """

      # Somewhat remarkably, when we create a Dir object here using '.', it
      # actually represents the directory containing the SConscript that called
      # this function.  This is handy, since we want to use that as the key in
      # the dictionary we're building.
      package_dir = env.Dir('.')
      _, dir_name = os.path.split(package_dir.get_abspath())
      if not files_for_doc_build.has_key(dir_name):
         files_for_doc_build[dir_name] = (pkg_title, categories, list())
   env.DocBuildThisPackage = DocBuildThisPackage
      

   # a function that wraps up the Pyc, Pyo, DocTest, and LogrefTest builders and
   # verifiers, and which enforces the dependency of the builds on the tools
   # files
   def PyFile(pyfile, logref=None, doctest_files=None, no_test=False,
              tool_file=False, no_doc=False, category=None):
      """
      Verify byte-compilation and testing for pyfile.

      pyfile should name a Python file that implements the Onyx doctesting
      facility via onyx.onyx_mainstartup.  The file will be compiled, its doctests
      will be run, and its test results will be evaluated.

      If logref is not None, pyfile should name a Python file that also
      implements the Onyx logreftest facility (via the --logreftest option),
      and logref should name a reference log file for the logref test.  The file
      will be compiled, its logref test be run, and its resulting log file will
      be compared with the reference file.

      If it is not None, doctest_files should be a list of files which are
      referenced in the doctests or logreftest of pyfile, which will then be
      added as dependencies.

      Returns a list that starts with the dependency node for the .pyc file
      corresponding to pyfile and which will include test output nodes too.

      If no_doc is False, then this file will be included in the automatic
      documentation generation done as part of the build.

      If category is not None, then it must be one of the strings used in
      DocBuildThisPackage as a category name.
      """

      # XXX it would be nice to allow pyfile to be a (list) of filenames or
      # nodes

      def build(pyfile):
         # explicit mention of the file to SCons
         pynode = env.File(pyfile)

         # global accumulation
         managedmodules.append(pyfile)

         # cause byte compiles to happen
         pycfile = env.Pyc(pyfile)
         pyofile = env.Pyo(pyfile)
         pycofiles = pycfile + pyofile
         #print 'pycofiles:', pycofiles

         # make sure targets get rebuilt if tools have changed
         if not tool_file:
            # tool_file is True for tools used by SCons in order to avoid
            # circular dependencies
            env.Depends(pycofiles, tool_files)

         # we make pyo_compile depend on all source files, so it doesn't execute
         # until all Pyo builders have executed (and their execution builds the
         # list of sources that need their PYOs built)
         env.Depends(pyo_compile, pynode)

         # accumulate into global containers used for doc generation (see
         # SphinxDocBuild)
         path, filename = os.path.split(pynode.get_abspath())
         path, package_name = os.path.split(path)
         path, location = os.path.split(path)
         if location == 'onyx' or location == 'py':
            # all pyc files under py become dependencies for sphinx
            nodes_for_doc_build_dependency.append(pycfile)

         if not no_doc:
            # For now we generate documentation only on the core modules (those in .../py/onyx)
            if (location == 'onyx' or location == 'py') and files_for_doc_build.has_key(package_name):
               # but py files have to be within a package that we're documenting
               assert len(files_for_doc_build[package_name]) == 3
               pkg_title, categories, module_file_list = files_for_doc_build[package_name]
               if categories is not None and category not in set((pair[0] for pair in categories)):
                  raise ValueError("Expected module %s to be in one of these categories: %s,"
                                   " but found category %s" % (filename, categories, category))
               module_file_list.append((category, filename))                  

         return pycofiles

      # hmmm, the Pyc and Pyo builders can do this themselves....
      if not pyfile.endswith('.py'):
         pyfile += '.py'

      # compile the py{co} files
      pyc_node = build(pyfile)[0:1]
      #print 'pyc_node:', ' '.join(str(x) for x in pyc_node)

      # note: create a copy so that we can append to exposed_targets without
      # messing up pyc_node
      exposed_targets = pyc_node[:]

      # we don't test if the PyFile() call said not to
      if not no_test:

         if doctest_files is not None:
            if not isinstance(doctest_files, (list, tuple)):
               doctest_files = [doctest_files]
         else:
            doctest_files = []

         # run the doctest, include doctest_files as sources so that the
         # scanner lists them in the build tree as dependencies
         doctest_nodes = env.DocTest(pyc_node + doctest_files)

         # XXX
         if True:# not tool_file:
            # tool_file is set for tools used by SCons in order to avoid
            # circular dependencies
            env.Depends(doctest_nodes, tool_files)

         #print 'doctest_nodes:', ' '.join(str(x) for x in doctest_nodes)
         exposed_targets += doctest_nodes

         # verify the doctest log
         env.DocTestVerify(doctest_nodes)         

         if logref is not None:
            logref_node = [env.File(logref)]

            # run the logreftest
            logreftest_nodes = env.LogrefTest(pyc_node)
            # this Depends may eliminate the need for the scanner in LogrefTest
            env.Depends(logreftest_nodes, doctest_nodes)
            #if doctest_files is not None:
            #   env.Depends(logreftest_nodes, doctest_files)

            # XXX
            if True:# and tool_file:
               # tool_file is set for tools used by SCons in order to avoid
               # circular dependencies
               env.Depends(logreftest_nodes, tool_files)

            #print 'logreftest_nodes:', ' '.join(str(x) for x in logreftest_nodes)
            exposed_targets += logreftest_nodes

            # verify the logreftest log
            env.LogrefTestVerify(logreftest_nodes + logref_node)

            #print 'exposed_targets:', ' '.join(str(x) for x in exposed_targets)

      return exposed_targets

   env.PyFile = PyFile


   def DocTestSummary():

      def summarize(env, target, source):
         totals = oks = bads = 0
         for total, ok, bad in doctestsummary:
            totals += total
            oks += ok
            bads += bad
         msg = "managed_modules %d  tested_modules %d  tested_statements %d  num_ok %d  num_bad %d" % (len(managedmodules), len(doctestsummary), totals, oks, bads)
         with open(str(*target), 'wt') as ofile:
            ofile.write(msg)
            ofile.write('\n')
         
      def cat(env, target, source):
         # copy singleton target to stdout
         from sys import stdout
         write = stdout.write
         with open(str(*target), 'rt') as infile:
            for line in infile:
               write(line)

      env.Command('doctest_summary', doctestsummary, [summarize, cat])

   env.DocTestSummary = DocTestSummary


   # This function, which is accessible as an attribute of the SCons
   # environment, will put a command in the environment that will build the HTML
   # documentation for the project using Sphinx.  The source_dir argument is
   # where to do the building; the aux_scons_sources is a list of other SCons
   # nodes that are needed for the build.
   def SphinxDocBuild(source_dir, aux_scons_sources, relative_src_dir):
      # print 'relative_src_dir', relative_src_dir
      # print "env['NO_DOC']", env['NO_DOC']
      if env['NO_DOC']:
         print "Not building documentation since NO_DOC is true"
         return
      
      _LEGAL_TYPES = set(('mod', 'cat', 'pkg'))
      def make_rst_filename(basename, type, package_name=None):
         assert(type in _LEGAL_TYPES)
         if package_name is not None:
            basename = package_name + '_' + basename
         return basename + '_' + type + '.rst'

      def write_core_rst_files(dest_directory):

         def write_rst_file(filename, contents):
            rst_filename = os.path.join(dest_directory, filename)
            with open(rst_filename, 'wb') as f:
               f.write(contents)

         for pkg_name, (pkg_title, categories, cat_fname_pairs) in files_for_doc_build.items():
            base_pkg_name = None if pkg_name == 'onyx' else 'onyx'
            if categories is not None:
               cat_names = tuple((pair[0] for pair in categories))
               pkg_contents = make_package_rst_contents(pkg_name, pkg_title, cat_names, 'cat', base_pkg_name)
               rst_filename = make_rst_filename(pkg_name, 'pkg')
               write_rst_file(rst_filename, pkg_contents)
               for cat_name, cat_doc in categories:
                  fnames_this_cat = [fname for cat, fname in cat_fname_pairs if
                                     cat == cat_name and fname != '__init__.py']
                  cat_contents = make_category_rst_contents(cat_name, cat_doc, fnames_this_cat, pkg_name)
                  rst_filename = make_rst_filename(cat_name, 'cat', package_name=pkg_name)
                  write_rst_file(rst_filename, cat_contents) 
            else:
               mod_names = [os.path.splitext(os.path.basename(fname))[0] for _, fname in cat_fname_pairs
                            if fname != "__init__.py"]
               pkg_contents = make_package_rst_contents(pkg_name, pkg_title, mod_names, 'mod', base_pkg_name)
               rst_filename = make_rst_filename(pkg_name, 'pkg')
               write_rst_file(rst_filename, pkg_contents)

            # The modules themselves don't know if the package uses categories or not
            cat_fname_pairs = ((cat, name) for (cat, name) in cat_fname_pairs if name != '__init__.py')
            for mod_category, mod_filename in cat_fname_pairs:
               assert(mod_filename != '__init__.py')
               mod_name, _ = os.path.splitext(os.path.basename(mod_filename))
               mod_contents = make_module_rst_contents(pkg_name, mod_name, base_pkg_name)
               rst_filename = make_rst_filename(mod_name, 'mod', package_name=pkg_name)
               write_rst_file(rst_filename, mod_contents)

      COMMENT = '.. THIS FILE WAS AUTOMATICALLY GENERATED - DO NOT EDIT\n'
      AUTOMODULE_OPTIONS = (':members:', ':inherited-members:', ':show-inheritance:', ':undoc-members:\n')
      def make_module_rst_contents(pkg_dir_name, mod_name, base_pkg_dir):
         prefix = base_pkg_dir + '.' if base_pkg_dir is not None else ''
         assert(mod_name != '__init__')
         module = '%s%s.%s' % (prefix, pkg_dir_name, mod_name)
         out_lines = [COMMENT]
         directive = '.. automodule::  %s' % (module,)
         out_lines.extend([directive])
         indent = '   ' # to line up under 'a' in 'automodule' 
         out_lines.extend((indent + opt for opt in AUTOMODULE_OPTIONS))
         return '\n'.join(out_lines)

      def make_category_rst_contents(cat_name, cat_doc, filenames, pkg_name):
         out_lines = [COMMENT]
         title = cat_name + ' Modules'
         under = '=' * len(title) + '\n'
         out_lines.extend((title, under))
         out_lines.extend((cat_doc + '\n',))
         directive = '.. toctree::'
         indent = '   ' # to line up under first 't' in 'toctree' 
         option = indent + ':maxdepth: 2\n'
         out_lines.extend([directive, option])
         for mod_filename in filenames:
            basename, _ = os.path.splitext(os.path.basename(mod_filename))
            rst_filename = make_rst_filename(basename, 'mod', package_name=pkg_name)
            out_lines.extend((indent + rst_filename,))
         return '\n'.join(out_lines)

      def make_package_rst_contents(pkg_name, pkg_title, names, type, base_pkg_dir):
         assert(type in _LEGAL_TYPES)
         out_lines = [COMMENT]
         indent = '   ' # to line up under the first letter in a directive

         doc_title = "%s" % (pkg_title,)
         under = '=' * len(doc_title) + '\n'
         out_lines.extend([doc_title, under])

         # Build top-level doc for this package from the package's __init__ file
         prefix = base_pkg_dir + '.' if base_pkg_dir is not None else ''
         directive = '.. automodule::  %s%s' % (prefix, pkg_name)
         out_lines.extend([directive])
         out_lines.extend((indent + opt for opt in AUTOMODULE_OPTIONS))

         # If there are no categories, add a section title between the
         # automodule and toctree directives to get proper indentation.  Note
         # that it's critical that we're using a different underline character
         # here than we used in the document title above.
         if type == 'mod':
            section_title = "%s Modules" % (pkg_title,)
            under = '+' * len(section_title) + '\n'
            out_lines.extend((section_title, under))

         # Now add a toctree with either the category rst files or the modules
         directive = '.. toctree::'
         option = indent + ':maxdepth: 2\n'
         out_lines.extend((directive, option))
         out_lines.extend((indent + make_rst_filename(name, type, package_name=pkg_name) for name in names))
         return '\n'.join(out_lines)

      def prepare_to_make_html_doc(env, target, source):
         # Note that we are relying on source_dir being in scope here
         print "Generating ReST files for project in %s " % (source_dir,),
         write_core_rst_files(source_dir)
         print " done"

      # We use the package rst files that this process will generate as a proxy
      # for the entire set of targets.  Since Sphinx has its own machinery for
      # deciding which html files to regenerate, we're not going to try to get
      # fancy and describe the entire dependency structure to SCons.
      rst_files = tuple([dir_name + '_pkg.rst' for dir_name, _ in files_for_doc_build.items()])

      doc_build_top = '_build'
      doc_build_doctrees = os.path.join(doc_build_top, 'doctrees')
      doctree_dir = os.path.join(source_dir, doc_build_doctrees)
      doc_build_html = os.path.join(doc_build_top, 'html')
      html_dir = os.path.join(source_dir, doc_build_html)

      sphinx_build_command = ' '.join(('sphinx-build',
                                       '-b', 'html',
                                       '-d', doctree_dir,
                                       source_dir,
                                       html_dir))

      env['SPHINX_BUILD_COMMAND'] = sphinx_build_command
      env['SPHINX_HTML_DIR'] = html_dir

      # for convenience, we create a symbolic link at the top of the
      # build-tree into the htlm documentation that sphinx builds
      link_source = os.path.join(str(env['ONYX_BUILD']), 'htmldoc')
      link_target = os.path.join(relative_src_dir, doc_build_html)
      link_cmd = 'ln -s %s %s' % (link_target, link_source)
      link_source2 = os.path.join(str(env['ONYX_BUILD']), 'htmldoc.html')
      link_target2 = os.path.join(relative_src_dir, doc_build_html, 'index.html')
      link_cmd2 = 'ln -s %s %s' % (link_target2, link_source2)

      # Put a build of the documentation on the list of things that (might) need to be done.
      rst = env.Command(rst_files, nodes_for_doc_build_dependency,
                        [prepare_to_make_html_doc, '${SPHINX_BUILD_COMMAND}'])
      env.Depends(rst_files, aux_scons_sources)
      env.Command(link_source, rst, [link_cmd])
      env.Command(link_source2, rst, [link_cmd2])


   # Put this function into the environment so some SConscript file can call it
   env.SphinxDocBuild = SphinxDocBuild

   def PylibSharedLibrary(module_shared_base, module_lib_shared=None):
      # copy the shared library to place(s) where Python's that have done
      # 'import onyx' will find it.
      env.Install(env.Dir(env['ONYX_BUILD_PYLIB']),
                  module_lib_shared if module_lib_shared is not None else env.File(env.make_pylib_name(module_shared_base)))
      # for standard build and an actually built shared library, copy it to the
      # source; use Install since Copy doesn't seem to work when copying back
      # into the source tree
      if module_lib_shared is not None:
         #print 'PylibSharedLibrary:', str(module_lib_shared[0]), '->', str(env.Dir('.').srcnode())
         env.Install(env.Dir('.').srcnode(), module_lib_shared)
   env.PylibSharedLibrary = PylibSharedLibrary

   def PylibSharedLibraryFile(filename):
      env.File(env.make_pylib_name(filename))
   env.PylibSharedLibraryFile = PylibSharedLibraryFile

   def PylibDepends(target, module_shared_base):
      # this tells SCons about the dependency of the Python module (and tests) on a shared library
      env.Depends(target, env.Dir(env['ONYX_BUILD_PYLIB']).File(env.make_pylib_name(module_shared_base)))
      #print 'PylibDepends:', str(target[0]), '->', str(env.Dir(env['ONYX_BUILD_PYLIB']).File(env.make_pylib_name(module_shared_base)))
   env.PylibDepends = PylibDepends

   def DataFile(filename):
      env.File(filename)
   env.DataFile = DataFile
      
   def source_file(target, source, env):
      target, = target
      source, = source
      pass

   # the doctest runner and verifier
   env.Append(BUILDERS = {
      'SourceFile': env.Builder(
      single_source = True,
      action = SCons.Action.Action(source_file, 'source_file( $SOURCE )'),
      ),
      })

   def PyPackageDepends(target, depends):
      depends = env.Flatten(depends)
      #print 'PyPackageDepends:', depends
      deps = set()
      for depend in depends:
         parts = depend.split('.')
         sep = package = ''
         for part in parts:
            package += sep + part
            sep = '/'
            deps.update(python_import_scanner(env.File(os.path.join(onyx_home, py_dir, package + '/__init__.pyc')), env, [os.path.join(onyx_home, py_dir)]))
      deps = sorted(deps)
      #print 'PyPackageDepends:', deps
      env.Depends(target, deps)
   env.PyPackageDepends = PyPackageDepends

   def PyModuleDepends(target, depends):
      depends = env.Flatten(depends)
      #print 'depends1:', depends
      deps = set()
      for depend in depends:
         deps.update(python_import_scanner(env.File(os.path.join(onyx_home, depend)), env, [os.path.join(onyx_home, py_dir)]))
      deps = sorted(deps)
      #print 'deps:', deps
      env.Depends(target, deps)
   env.PyModuleDepends = PyModuleDepends



def exists(env):
   """
   Make sure that PyTools has the resources it needs.
   """
   if not __debug__:
      # removing this restriction would require work when compiling PYC and PYO files
      print >> sys.stderr, debug_only
      return False
   else:
      return True

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
