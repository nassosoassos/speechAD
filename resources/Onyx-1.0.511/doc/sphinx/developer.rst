..
 ==========================================================================
 =
 = File:         developer.rst
 = Date:         11-Aug-2009
 = Author:       Hugh Secker-Walker
 = Description:  Documentation for installing the project for development work
 =
 = This file is part of Onyx   http://onyxtools.sourceforge.net
 =
 = Copyright 2009 The Johns Hopkins University
 =
 = Licensed under the Apache License, Version 2.0 (the "License").
 = You may not use this file except in compliance with the License.
 = You may obtain a copy of the License at
 =   http://www.apache.org/licenses/LICENSE-2.0
 = 
 = Unless required by applicable law or agreed to in writing, software
 = distributed under the License is distributed on an "AS IS" BASIS,
 = WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 = implied.  See the License for the specific language governing
 = permissions and limitations under the License.
 =
 ==========================================================================


======================
Developer Installation
======================

These instructions are for users users of Onyx who want to work with and modify
the source code of the project.


This file is the top-level readme for Onyx.  It gives quick instructions about
how to retrieve a copy of the project, and it points to other documentation
within the project.  These instructions should get you going on either the
HLTCOE HPCC machines, on a Mac, or on Linux.

.. note::

   some of the instructions below are out of date.


Preparing to install the project
================================

The project uses the Bazaar version control system.  The Bazaar
version control is often called bazaar or bzr and is also sometimes
referred to as bazaar-ng (for next generation).  See:
http://bazaar-vcs.org/
You will need to install bazaar before you can retrieve the HLTCOE
project.


Installing Bazaar on HPCC
-------------------------

The HPCC machines have Bazaar already installed in /usr/local/bin.  Try "which
bzr" and verify that you get "/usr/local/bin/bzr".  Then you can move on to
"Installing the project" below.

Installing Bazaar on Mac
------------------------

On the Mac platform we recommend using the command-line program 'port' to
retrieve and build OpenSource projects used by HLTCOE.  To prepare your system
for :command:`port`, you need to install `Apple's Xcode SDK
<http://developer.apple.com/tools/xcode/>`_.  Xcode is both an application and
an SDK toolchain.  It contains GCC tools which are needed by the :command:`port`
command when it builds packages.

Once you have Xcode installed, then install the `MacPorts package
<http://www.macports.org/>`_.  This installs tools under the /opt/local
directory.  It may also fiddle your path, e.g. in :file:`~/.profile`.

MacPorts has command-line tools that let you manage the installation
of numerous OpenSource projects.  The main tool is :command:`port`::

  $ port help
  port
          [-bcdfiknopqRstuvx] [-D portdir] [-F cmdfile] action [privopts] [actionflags]
          [[portname|pseudo-portname|port-url] [@version] [+-variant]... [option=value]...]...

  Supported commands
  ------------------
  activate, archive, build, cat, cd, checksum, clean, compact, configure,
  contents, deactivate, dependents, deps, destroot, dir, distcheck, distfiles,
  dmg, dpkg, echo, ed, edit, exit, extract, fetch, file, gohome, help,
  info, install, installed, lint, list, livecheck, load, location, mdmg,
  mirror, mpkg, outdated, patch, pkg, platform, provides, quit, rpm, search,
  selfupdate, srpm, submit, sync, test, trace, unarchive, uncompact, uninstall,
  unload, upgrade, url, usage, variants, version, work

  ...  [ for many lines ]

  For more information
  --------------------
  See man pages: port(1), macports.conf(5), portfile(7), portgroup(7),
  porthier(7), portstyle(7).

Use :command:`port` to install Bazaar version control.  Beware of a potential
confusion: the unrelated GNU Arch project is called bazaar in the MacPorts
databases; be sure to get bzr.  In order to use :command:`port` you must be
root, e.g. via :command:`sudo`.  Using the following you will be prompted for
the administrative password, then :command:`port` will get to work::

  $ sudo port install bzr

Note that :command:`port` manages package dependencies and will have to download
and build several other packages before it finishes.  To see more about what
it's doing you can try verbose mode or even debug mode::

  $ sudo port -v install bzr
  $ sudo port -d install bzr

When :command:`port` has finished its work, you should be able to run bzr in a
(new) shell.  E.g.::

  $ bzr
  Bazaar -- a free distributed version-control tool
  http://bazaar-vcs.org/

  Basic commands:
    bzr init           makes this directory a versioned branch
    bzr branch         make a copy of another branch

    bzr add            make files or directories versioned
    bzr ignore         ignore a file or pattern
    bzr mv             move or rename a versioned file

    bzr status         summarize changes in working copy
    bzr diff           show detailed diffs

    bzr merge          pull in changes from another branch
    bzr commit         save some or all changes

    bzr log            show history of changes
    bzr check          validate storage

    bzr help init      more help on e.g. init command
    bzr help commands  list all commands
    bzr help topics    list all help topics




Installing the project
======================

The project is accessed from a repository on an HLTCOE server.

Access to the server
--------------------

At present you need an account on the HLTCOE network in order to
retrieve a copy of the project.  You must also be a member the
**bzruser** group.  The IT folks can set you up with an account.

When accessing the server, it's a huge convenience to use the
:command:`ssh` RSA authentication keys so that you don't have to type
your password each time you access the repository.  See the
:command:`ssh-keygen` man page.  On your local machine use
:command:`ssh-keygen` to generate the :file:`~/.ssh/id_rsa` and
:file:`~/.ssh/id_rsa.pub` files.  The following examples show this
process, where *<user>* should be replaced with your user name.  Be
sure to use an empty passphrase, otherwise you will be prompted for
the passphrase each time you try to access the server.

::

  $ ssh-keygen
  Generating public/private rsa key pair.
  Enter file in which to save the key (/home/<user>/.ssh/id_rsa):
  Enter passphrase (empty for no passphrase):

  Enter same passphrase again:

  Your identification has been saved in /home/<user>/.ssh/id_rsa.
  Your public key has been saved in /home/<user>/.ssh/id_rsa.pub.
  The key fingerprint is:
  b6:a8:2a:a9:c7:5e:bb:a6:ad:08:03:73:e8:ed:2d:17 <user>@hltcoe.hodain.net
  The key's randomart image is:
  +--[ RSA 2048]----+
  |                 |
  | .               |
  |+ .  E           |
  |ooo +. .         |
  |.. +.o  S        |
  |. ..o. o .       |
  |+.. . . .        |
  |oooo.o .+        |
  |oo+++            |
  +-----------------+

Then,
append the contents of the local :file:`~/.ssh/id_rsa.pub` to the
server file :file:`~/.ssh/authorized_keys`::

  $ ssh <user>@external.hltcoe.jhu.edu 'cat >> ~/.ssh/authorized_keys' < .ssh/id_rsa.pub
  <user>@external.hltcoe.jhu.edu's password:

Then verify that you can access the account and run a command using :command:`ssh`, e.g.::

  $ ssh <user>@external.hltcoe.jhu.edu hostname
  gpsrv5


Creating your local branch
--------------------------

Once you are set up to access the server, you can
create a local branch of the project.

This will prompt you for a password (on the HPCC machines you may get
prompted twice; this is OK) and then retrieve the project files,
putting them into the directory my_onyx_code.  Note that

You would use the following command with your user name in the
*<user>* slot.  This step may take a while to finish without giving
any indication that anything is happening.

::

  $ bzr branch bzr+ssh://<user>@external.hltcoe.jhu.edu/srv/bzr/code my_onyx_code

This will have installed a local :term:`branch` of the project in the
:file:`my_onyx_code` directory.

The URL :file:`bzr+ssh://<user>@external.hltcoe.jhu.edu/srv/bzr/code`
refers to what is called the :term:`parent` repository.  It's the
version of the project (and all its history) from which the local
branch is created.

The local branch consists of a :term:`working tree` and a
:term:`repository`.  The working tree is all of the files and data of
the project, it's what you see if you browse around in
:file:`my_onyx_code`.  This is where you look at code and modify files, etc.

A repository is where :command:`bzr` does all
its recordkeeping.  The local repository is found under
:file:`my_onyx_code/.bzr`.

.. note::

  Do not change anything in the repository; it is the :command:`bzr` database.



Working with your local branch
==============================

The project has several commands that use bzr tools to manage you
local branch and the parent repository.  These commands live in the
:file:`bin` directory.  They are typically executed from the top-level
directory of the project.  This convention makes it easy to work with
several different installations of the project.

To get a summary of what's new in the repository since you last
synchronized with the repository.  The :command:`snew` command shows
you the differences between the repository's change file, which is
:file:`changes.txt`.  If you are up to date you will get a short
message to that effect::

  $ bin/snew
  Local revision is 461, which is current for parent: bzr+ssh://<user>@external.hltcoe.jhu.edu/srv/bzr/code/
  You are up to date!

If you are out of date you will get a more verbose message::

  $ bin/snew
  Diffing ./changes.txt with parent version at bzr+ssh://<user>@external.hltcoe.jhu.edu/srv/bzr/code/
  === modified file 'changes.txt'
  *** local changes.txt   2009-07-22 16:20:32 +0000
  --- parent changes.txt  2009-08-11 18:46:20 +0000
  ***************
  *** 15 ****
  --- 16,112 ----
  + Commit at 2009-08-11 14:46:19 -0400 by <username> <<user>@<host>> from <branch-nick>
  + Message: Additions to top-level documentation
  + Last revno: 460   Last revision id: <revision-id>
  + Last revision date: 2009-08-11 13:26:33 -0400
  +
  + modified:
  +   doc/sphinx/SConscript
  +   doc/sphinx/index.rst
  +   doc/sphinx/toplevelorg.rst
  + added:
  +   doc/sphinx/developer.rst
  +   doc/sphinx/devo
  +   doc/sphinx/user.rst
  +   templates/restructured.rst
  +     Updates to the documentation top-level structure.


There is also a less-informative lower-level command that shows you
bzr's view of what is different between your branch and its parent
repository.  You use :command:`missing` to see what :command:`merge`
or :command:`pull` would retrieve from the parent repository in order
to bring your local branch up to date with the parent repository::

  $ bin/missing
  Using saved parent location: bzr+ssh://<user>@external.hltcoe.jhu.edu/srv/bzr/code/
  You are missing 1 revision(s):
  ------------------------------------------------------------
  revno: 461
  committer: <username> <<user>@<host>>
  branch nick: <branch-nick>
  timestamp: Tue 2009-08-11 14:46:20 -0400
  message:
    Additions to top-level documentation

When you are out of date, and you want to update, use the
:command:`pull` command to retrieve updated versions of files from the
server::

  $ bin/pull
  Using saved parent location: bzr+ssh://<user>@external.hltcoe.jhu.edu/srv/bzr/code/
  +N  doc/sphinx/developer.rst
  +N  doc/sphinx/devo
  +N  doc/sphinx/user.rst
  +N  templates/restructured.rst
   M  changes.txt
   M  doc/sphinx/SConscript
   M  doc/sphinx/index.rst
   M  doc/sphinx/toplevelorg.rst
  All changes applied successfully.
  Now on revision 461.


You can also use :command:`bin/merge`, which is like the pull command,
but is used when there are change conflicts that need to be merged,
and which may require your intervention to resolve.



Making changes
--------------

To make changes to the project you just modify the files in question.
In order to add a new file or directory to the project, use the
:command:`bzr add` command.

To see how :command:`bzr` views the status of the project, use the
:command:`status` command::

  $ bin/status
  added:
    LICENSE.txt
    NOTICE.txt
  modified:
    doc/presentations/logo-samples.ppt
    py/onyx/__init__.py
  unknown:
    LICENSE.BSD.txt
    usage-example.txt
    doc/work/

This is a complex example:

* some files that have been **added** to the project, but have not yet been checked in
* some project files have been **modified**
* some files and directories are **unknown** as far as :command:`bzr` is concerned



Checking-in your changes
------------------------

Once you have a coherent set of changes that are tested and
documented, and the build succeeds, then you are ready to check-in.

We use a small set of commands that we layer on top of Bazaar.  The
purpose of these commands is to automatically update the project file
:file:`changes.txt` which lives in the top-level of the project.  Take
a look at that file to see a prose history (most recent first) record
of changes to the project.

If you have made changes to files in the project which you are ready
to commit, here's the procedure you should follow.  Begin by doing::

  $ bin/prep
  modified:
    doc/presentations/logo-samples.ppt
    py/onyx/__init__.py
  added:
    LICENSE.txt
    NOTICE.txt

   ===== Additional bzr output - FYI only =====

  unknown:
    LICENSE.BSD.txt
    usage-example.txt
    doc/work

  Renamed /Users/hugh/onyx/onyx_doc/comments.txt to /Users/hugh/onyx/onyx_doc/comments.bak
  Wrote new template file: /Users/hugh/onyx/onyx_doc/comments.txt
  Wrote new checkin file: /Users/hugh/onyx/onyx_doc/ckin_files.py
  Now edit comments.txt then run ckin to complete your commit
  You may also edit ckin_files.py if you wish to exclude some modified files from this check-in.

This will fill out the file :file:`comments.txt` in the top level of your
project with a template showing which files you've changed, added, removed, etc,
based on the output of :command:`bin/status`.

Now edit :file:`comments.txt`, adding text describing what you've done.
Descriptions generally run from a one or two sentences to a few paragraphs, but
occasionally longer entries are appropriate.  See the project
:file:`changes.txt` for examples of the descriptive prose style you should add
to :file:`comments.txt`.

When you are done editing your descriptions of the work you've done to the
project it's a very good idea to run SCons one more time, just in case you've
inadvertently changed something.  If you have not changed anything this step
doesn't take long::

  $ bin/scons
  scons: Reading SConscript files ...
  scons: done reading SConscript files.
  scons: Building targets ...
  scons: building associated BuildDir targets: build
  scons: `.' is up to date.
  scons: done building targets.


Now you are ready to check in::

  $ bin/ckin '<log message>'

where *<log message>* is a short message for the bzr log.  This command will
automatically insert the contents of :file:`comments.txt` at the top of
:file:`changes.txt`, and then it will commit the changes from your working tree
to your local repository.  This latter step is done via :command:`bzr`'s
:command:`commit` machinery.

At this point, your local branch should have no status::

  $ bin/status


When it's time to propogate your commited changes to the parent
repository, so that everyone else can retrieve them, do::

  $ bzr push

This updates the server's branch so that it is a mirror of your local branch.
Once you've done this, when other developers of the project do
:command:`bin/snew`, they will see you description of your changes, and they'll
do :command:`bin/pull`, or :command:`bin/merge` to propogate those changes into
their local branch.


Documentation in the project
============================

Top-level directories in the project often have a readme.  Here are
some of them:

*  ./bin/README.txt
*  ./cpp/matrix/README.txt
*  ./py/README.txt         important pointers for building and extending
*  ./README.txt
*  ./scons/scons-README
*  ./templates/README.txt  templates with boilerplate for various kinds of files

The directory ./readme has some documentation that doesn't seem to belong
anywhere else, e.g. how to set up a Mac for the project.


Preparing to build the project
==============================

In order to build and use the project, you will need some third-party tools.
Here's a list of what's known to be needed.

|  Python 2.5
|  Python packages:
|    numpy
|    scipy
|
|  C++ compilation tools


Preparing to build on HPCC
--------------------------

Everything you need to build on the HPCC machines is already installed and
should be on your path.


Preparing to build on the Mac
-----------------------------

The Xcode toolchain is required.  See:
http://developer.apple.com/tools/xcode/


Python is retrieved, built, and installed via :command:`port`.  The list of
available packages is huge (over 4000) so be selective when searching.
E.g. there over 85 Python 2.5 related sub-packages::

  $ port search py25 | wc
  89

Here is a recommended set of Python tools::

  $ sudo port -v install python25 py25-ipython py25-numpy

This will build and test numerous components and it takes a while to
run (go get lunch).


Build Environment
=================

We are trying to keep the number of environment variables you must set to a
minimum.  You should put the 'py' directory of the project on your
:envvar:`PYTHONPATH`.  E.g. if you've installed the project in
:file:`~/my_onyx_code`, then you would have the following in your
:file:`./profile` or equivalent::

    # Setting the path for python and MacPorts
    export PYTHONPATH=~/my_onyx_code/py

Setting up the build environment on HPCC
----------------------------------------

You shouldn't need to set any other environment variables on the HPCC.

Setting up the build environment on Mac
---------------------------------------

You should verify that the :command:`port` bin directories are on your
:envvar:`PATH`.  You should have the following in your
:file:`./profile` or equivalent::

    # Setting the path for MacPorts
    export PATH=/opt/local/bin:/opt/local/sbin:$PATH


Building the project
====================

Onyx is a build-based platform.  That is, before using the tools you have to
build and test them.  The building is handled by the SCons software construction
tool.  A version of this tool is included in the project so you do not have to
install it.  The build and test process is unified.  That is, when SCons is
activated it only builds and tests those parts of the project that are out of
date.

Scons documentation is online.  The `SCons Tutorial
<http://www.scons.org/doc/1.2.0/HTML/scons-user/book1.html>`_ is
useful as a starting point, but not as a reference.  The `SCons Man
Page <http://www.scons.org/doc/1.2.0/HTML/scons-man.html>`_ is a very
useful reference once you have learned the basic ideas.

If all is well in your project, you should be able to run SCons at the top level
and it will go and build and test everything.  By default it stops as soon as it
encounters an error.  A successful build looks like this::

  > bin/scons
  scons: Reading SConscript files ...
  scons: done reading SConscript files.
  scons: Building targets ...
  scons: building associated BuildDir targets: build
  g++ -o build/cpp/liveaudio/DCAudioFileRecorder.o -c build/cpp/liveaudio/DCAudioFileRecorder.cpp
  g++ -o build/cpp/liveaudio/audio.o -c build/cpp/liveaudio/audio.cpp
  ar rc build/cpp/liveaudio/libaudio.a build/cpp/liveaudio/audio.o build/cpp/liveaudio/DCAudioFileRecorder.o
  ranlib build/cpp/liveaudio/libaudio.a

  ... [ for many lines ]

  python -O -m onyx.onyx_py_compile build/templates/__init__.py
  python -m onyx.onyx_py_compile build/templates/module.py
  python build/templates/module.pyc > build/templates/module.log-doctest
  verify_log_doctest( build/templates/module.log-doctest > build/templates/module.log-doctestverify )
  python -O -m onyx.onyx_py_compile build/templates/module.py
  summarize(["doctest_summary"], [])
  cat doctest_summary
  managed_modules 36  tested_modules 0  tested_statements 0  num_ok 0  num_bad 0
  scons: done building targets.

.. note::

   If the build is successful the SCons output will finish with the
   line ``scons: done building targets.`` If the build failed then the
   SCons output will end with something like ``scons: building
   terminated because of errors.`` or ``scons: done building targets
   (errors occurred during build).``


If there is no work to be done when you run SCons you will get the following::

  $ bin/scons
  scons: Reading SConscript files ...
  scons: done reading SConscript files.
  scons: Building targets ...
  scons: building associated BuildDir targets: build
  scons: `.' is up to date.
  scons: done building targets.

Usually, when you've modified code, some subset of the code will get rebuilt and
retested.

As you can see from the log, SCons is configured so that all the output of the
build process goes into the :file:`build/` subdirectory in the project.  Within
this subdirectory there will be a platform-specific subdirectory, e.g.
:file:`build/darwin-posix-i386-32bit-le/`.  The output of the build process goes
into this platform-specific subdirectory.  This allows you to build from
multiple platforms that share the same local branch.

To remove everything that gets built by SCons use the
:command:`-c` command-line option::

  $ bin/scons -c

When there is a problem during the build SCons stops immediately.  If you'd like
SCons to continue despite errors, use the :command:`-k` command-line option::

  $ bin/scons -k

SCons is threadsafe.  If you have multiple cores on your machine you can get a
significant speedup of the build by telling SCons how many jobs it can run
simultaneously.  Empirically, we've observed that specifying 1.5 to 2 times as
many cores as you have gives the fastest builds.  E.g. on a 2-core system::

  $ bin/scons -j 4

Note that the outputs of multiple simultaneously running jobs will be
interleaved.

To stress the build process, you can tell SCons to randomly select the order in
which it will run jobs (while still respecting the underlying dependencies of
the build)::

  $ bin/scons -j 4 --random



TTD:

* Overview
* Notes specific to desktop Linux
* Nice hand-holding walk through
* Demos - scripts/Malach/social-network/sigproc/live-demo/training/decoding/sandbox/eventdet
