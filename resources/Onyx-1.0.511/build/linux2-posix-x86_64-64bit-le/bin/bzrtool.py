###########################################################################
#
# File:         bzrtool.py
# Date:         Wed 20 Feb 2008 15:26
# Author:       Ken Basye
# Description:  Wrapper around some bzr and other functionality to support project prep/ckin/snew operations
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

from __future__ import with_statement

import os, sys, tempfile, shutil
from time import strftime
from cStringIO import StringIO
from itertools import izip, repeat

import bzrlib
from bzrlib import trace
from bzrlib import bzrdir
from bzrlib import workingtree
from bzrlib import diff
from bzrlib import errors

COMMENT_FILE_NAME = 'comments.txt'
BACKUP_FILE_NAME = 'comments.bak'
CHANGES_FILE_NAME = 'changes.txt'
INSERTION_MARKER = '$$$$_MARKER_FOR_INSERTIONS_$$$$'
CKIN_FILE_NAME = 'ckin_files.py'
CKIN_TUPLE_NAME = 'ckin_files'
# We list these in the order we'd like to see them, roughly in order of importance
STATUS_TYPES = ('conflicts', 'pending_merges', 'modified', 'added', 'removed', 'renamed', 'kind_changed', 'unknown')

class BzrToolError(StandardError):
    def __init__(self, report):
        self.error_report = report

class BzrToolTransportError(BzrToolError):
    def __init__(self, report):
        super(BzrToolTransportError, self).__init__(report)

def _collect_tree_status(tree):
    status_dict = dict()
    for st in STATUS_TYPES:
        status_dict[st] = None
    if tree.last_revision() != tree.branch.last_revision():
        warning("working tree is out of date, run 'bzr update'")
    old = tree.basis_tree()

    old.lock_read()
    tree.lock_read()
    try:
        # We can get most types of changes from the TreeDelta between our working tree and the basis.
        delta = tree.changes_from(old, want_unversioned=True)
        delta.unversioned = [unv for unv in delta.unversioned if not tree.is_ignored(unv[0])]
        status_dict['modified'] = tuple(delta.modified) if len(delta.modified) > 0 else None
        status_dict['added'] = tuple(delta.added) if len(delta.added) > 0 else None
        status_dict['renamed'] = tuple(delta.renamed) if len(delta.renamed) > 0 else None
        status_dict['removed'] = tuple(delta.removed) if len(delta.removed) > 0 else None
        status_dict['unknown'] = tuple(delta.unversioned) if len(delta.unversioned) > 0 else None
        status_dict['kind_changed'] = tuple(delta.kind_changed) if len(delta.kind_changed) > 0 else None
        
        # Conflicts and pending merges have to be gotten separately
        conflicts = tree.conflicts()
        if len(conflicts) > 0:
            status_dict["conflicts"] = tuple(conflicts)

        # Pending merges are complicated, so we use the status module to get a
        # complete text description and put that in a tuple in the result.
        if len(tree.get_parent_ids()) > 1:
            strfile = StringIO()
            pos = strfile.tell()
            bzrlib.status.show_pending_merges(tree, strfile)
            strfile.seek(pos)
            contents = []
            for i, line in enumerate(strfile.readlines()):
                # skip the first line, which just has "pending merges:"
                if i == 0:
                    continue
                contents.append(line)
            status_dict['pending_merges'] = (("\n".join(contents),),)
    finally:
        old.unlock()
        tree.unlock()
    return status_dict

def _format_entry_for_comments(key, entry):
    assert(key != 'conflicts')
    if key == 'pending_merges':
        return '  ' + entry[0] + '\n'
    elif key == 'renamed':
        return '  ' + entry[0] + ' ==> ' + entry[1] + '\n'
    else: 
        return '  ' + entry[0] + '\n'

def _format_entry_for_ckin_file(key, entry):
    assert(key != 'conflicts' and key != 'pending_merges' and key != 'unknown')
    if key == 'renamed':
        return '    "' + entry[1] + '",\n'
    return '    "' + entry[0] + '",\n'


def _write_comments_file(status_dict, outfile):
    fyi_done = False
    for key in STATUS_TYPES:
        if status_dict[key] is None:
            continue
        if key == 'unknown' and not fyi_done:
            outfile.write('\n\n ===== Additional bzr output - FYI only ===== \n\n')
            print('\n\n ===== Additional bzr output - FYI only ===== \n\n')
            fyi_done = True
        line = key + ':\n'
        outfile.write(line)
        print(line.rstrip())    
        for entry in status_dict[key]:
            line = _format_entry_for_comments(key, entry)
            outfile.write(line)
            print(line.rstrip())    


def _write_ckin_file(status_dict, outfile):
    
    outfile.write("# This file was automatically written by 'prep'.\n")
    if status_dict['pending_merges'] is not None:
        outfile.write("# DO NOT EDIT - partial checkins are not allowed when merges are pending\n\n")
    else:
        outfile.write("# You may remove files you don't want checked in from the tuple below.\n\n")

    outfile.write('%s  = (\n' % CKIN_TUPLE_NAME)
    ckin_status_types = STATUS_TYPES[2:-1]
    for key in ckin_status_types:
       if status_dict[key] is None:
           continue
       line = '    # ' + key + ':\n'
       outfile.write(line)
       for entry in status_dict[key]:
           line = _format_entry_for_ckin_file(key, entry)
           outfile.write(line)
    outfile.write(')\n')


def _make_changes_header(tree, message):
    # Find name and date to go on the first line
    gconf = bzrlib.config.GlobalConfig()
    name = gconf.username()
    time = strftime("%Y-%m-%d %H:%M:%S %z") 
    # Long revision id, e.g., 'hsw@hodain.net-20080225223544-6gi784map4b1oeta'
    revid = tree.last_revision()
    # Short revno, e.g., '155'
    revno = tree.branch.revision_id_to_revno(revid)
    # Branch nickname
    nick = tree.branch.nick
    # Time of last revision
    rev = tree.branch.repository.get_revision(revid)
    timestamp = rev.timestamp
    timezone = rev.timezone
    revtime = bzrlib.osutils.format_date(timestamp, timezone, date_fmt = '%Y-%m-%d %H:%M:%S',
                                         timezone = 'original', show_offset = True)

    header = ("\n" + "_"*80 + "\n" + "Commit at " + time +  " by " + name + " from " + nick + "\n")
    header += "Message: " + message + "\n"   
    header += "Last revno: %d   " % (revno,)
    header += "Last revision id: " + revid + "\n"
    header += "Last revision date: " + revtime + "\n\n"
    return header
    
def _update_changes_file(srcname, destname, tree, message):
    header = _make_changes_header(tree, message)
    tempfd, tempname = tempfile.mkstemp()
    temp = os.fdopen(tempfd, "w")
    found_marker = False
    with open(destname) as dest:
        for line in dest:
            temp.write(line)
            if line.strip() == INSERTION_MARKER:
                found_marker = True
                temp.write(header)
                with open(srcname) as src:
                    for srcline in src:
                        temp.write(srcline)
    temp.close()
    if not found_marker:
        os.remove(tempname)
        raise BzrToolError("No insertion marker found in %s!" % (destname,))
    else:
        shutil.move(tempname, destname)
            
def get_local_info():
    log_file = StringIO()
    try:
        # Establish the top-level directory for this installation and get bzr WorkingTree
        logger = bzrlib.trace.push_log_file(log_file)
        local_tree, top = get_working_tree(os.getcwd())
        local_branch = local_tree.branch
        local_revno = local_branch.revno()
        local_last_revid = local_tree.last_revision()
        return (local_tree, local_revno, local_last_revid)
    except BzrToolError, e:
        mesg = e.error_report + "\n"
        if log_file.getvalue().split():
            mesg += "bzr also collected this logging information:\n"
            mesg += log_file.getvalue()
            mesg += "\n"
        raise BzrToolError(mesg)


def _get_local_and_parent_info():
    log_file = StringIO()
    try:
        # Establish the top-level directory for this installation and get bzr WorkingTree
        logger = bzrlib.trace.push_log_file(log_file)
        local_tree, top = get_working_tree(os.getcwd())
        local_branch = local_tree.branch
        local_revno = local_branch.revno()
        local_last_revid = local_tree.last_revision()
        parent_url = local_branch.get_parent()

        # This function wants to log some stuff, and it complains if the log
        # handler it expects to talk to hasn't been set up, which is why we give
        # it one using push_log_file above.
        parent_bzrdir, parent_relpath = bzrdir.BzrDir.open_containing(parent_url)

        parent_branch = parent_bzrdir.open_branch()
        assert parent_branch is not None
        # XXX not sure what basis_tree is doing exactly; I got this from bzrlib.diff
        parent_tree = parent_branch.basis_tree()
        parent_revno = parent_branch.revno()
        parent_last_revid = parent_branch.last_revision()

        # Figure out the status of the local and parent branches.  There are
        # four possible states: either of the local branch or the parent branch
        # may have new revisions, both may, or neither may.
        state = None
        if local_last_revid == parent_last_revid:
            state = "no_new_revisions"
        else:
            local_branch.lock_read()
            parent_branch.lock_read()
            local_heads = local_branch.repository.get_graph().heads([local_last_revid, parent_last_revid])
            parent_heads = parent_branch.repository.get_graph().heads([local_last_revid, parent_last_revid])
            local_branch.unlock()
            parent_branch.unlock()
            assert local_last_revid in local_heads
            assert parent_last_revid in parent_heads
            parent_new = parent_last_revid in local_heads
            local_new = local_last_revid in parent_heads
            if local_new and parent_new:
                state = "both_new_revisions"
            elif parent_new and not local_new:
                state = "parent_new_revisions"
            elif local_new and not parent_new:
                state = "local_new_revisions"
        assert state is not None
        bzrlib.trace.pop_log_file(logger)
        return (local_tree, local_revno, local_last_revid,
                parent_tree, parent_revno, parent_last_revid, parent_url, state)
    except BzrToolError, e:
        mesg = e.error_report + "\n"
        if log_file.getvalue().split():
            mesg += "bzr also collected this logging information:\n"
            mesg += log_file.getvalue()
            mesg += "\n"
        raise BzrToolError(mesg)
    except bzrlib.errors.TransportError, e:
        mesg = str(e) + "\n"
        if log_file.getvalue().split():
            mesg += "bzr also collected this logging information:\n"
            mesg += log_file.getvalue()
            mesg += "\n"
        raise BzrToolTransportError(mesg)
    

def check_up_to_date():
    mesg = ''
    result = _get_local_and_parent_info()

    assert len(result) == 8
    _, local_revno, local_last_revid, _, parent_revno, parent_last_revid, parent_url, state = result

    offer_bailout = False
    if state == 'parent_new_revisions':
        mesg += "You are NOT up to date!!\n"
        mesg += "Local revision is %d, which is NOT current for parent: %s\n" % (local_revno, parent_url)        
        mesg += "This means you almost certainly want to pull before you check in.\n"
        offer_bailout = True
    elif state == 'local_new_revisions':
        mesg += "You appear to have made previous commits in this tree.\n"
        mesg += "Local revision is %d, which is higher than parent: %s\n" % (local_revno, parent_url)
        mesg += "You may want to push sometime soon.\n"
    elif state == "both_new_revisions":
        mesg += "You are NOT up to date!!\n"
        mesg += "And you appear to have made previous commits in this tree\n"
        mesg += "Since the last revision ID in your local tree:\n"
        mesg += "   " + local_last_revid + "\n"
        mesg += "Doesn't match the last revision ID in your parent tree:\n"
        mesg += "   " + parent_last_revid + "\n"
        mesg += "You may want to consider running 'bzr uncommit' followed by 'bzr revert changes.txt'"
        mesg += "to undo your last commit.\n"
        offer_bailout = True

    print mesg
    if offer_bailout:
        reply = raw_input("Do you want to bail out now? [y/n]: ")
        while reply != 'y' and reply != 'n':
            reply = raw_input("Please answer 'y' or 'n': ")
        assert reply == 'y' or reply == 'n'
        return reply == 'y'  # Yes to bailing out
    else:
        return False
        

def get_working_tree(from_dir):
    # This handy function runs up the tree from the argument until it finds a
    # bzr root, then returns the working tree for that root.  With no argument
    # it starts from the cwd, but we want that value explicitly for other
    # purposes anyway.
    try:
        tree, relpath = bzrlib.workingtree.WorkingTree.open_containing(from_dir)
    except bzrlib.errors.NotBranchError, e:
        raise BzrToolError("No bzr root found moving up from %s" % (from_dir,))
    return tree, os.path.normpath(os.path.join(from_dir, relpath))

def prep(args):
    check_snew = True
    for arg in args:
        if arg == "--no_check":
            check_snew = False
        else: # unknown arg
            return False

    mesg = ""

    if check_snew:
        bailout = False
        try:
            bailout = check_up_to_date()
        except BzrToolTransportError, e:
            print e.error_report
            print "It looks like you are having problems getting to the parent for this installation"
            print "(maybe your network connection is down?).  You can re-run with '--no_check' to avoid"
            print "requiring a connection to the parent."
            return True
        except BzrToolError, e:
            print e.error_report
            return True
        if bailout:
            print "Bailing out now"
            return True

    # Establish the top-level directory for this installation and get bzr WorkingTree
    tree, top = get_working_tree(os.getcwd())

    # Collect status information, bail if we find conflicts
    status_dict = _collect_tree_status(tree)
    if status_dict['conflicts'] is not None:
        mesg += "Your installation has unresolved conflicts.  Please run 'bzr status' to\n"
        mesg += "identify this problem and then correct it.  No files were written for this prep.\n"
        print mesg
        return True

    # Check for existence of comments file and back up if necessary
    new_comments_name = os.path.join(top, COMMENT_FILE_NAME)
    backup_comments_name = os.path.join(top, BACKUP_FILE_NAME)
    if os.path.exists(new_comments_name):
        shutil.move(new_comments_name, backup_comments_name)
        mesg += "Renamed %s to %s\n" % (new_comments_name, backup_comments_name)

    # Write new file
    with open(new_comments_name, 'w') as outfile:
        _write_comments_file(status_dict, outfile)  # This will also print the file to the screen!
    mesg += "Wrote new template file: %s\n" % (new_comments_name,)

    # Bazaar doesn't support checking in specific files when there are pending merges,
    # but we write the file anyway for possible future checking.
    ckin_file_name = os.path.join(top, CKIN_FILE_NAME)
    with open(ckin_file_name, 'w') as outfile:
        _write_ckin_file(status_dict, outfile)
    mesg += "Wrote new checkin file: %s\n" % (ckin_file_name,)

    # Make announcements
    mesg += ("Now edit %s then run ckin to complete your commit\n" % (COMMENT_FILE_NAME,))
    if status_dict['pending_merges'] is not None:
        mesg += ('DO NOT EDIT %s - you have pending merges in this check-in.\n' %
                      (CKIN_FILE_NAME,))
    else:
        mesg += ('You may also edit %s if you wish to exclude some modified files from this check-in.\n' %
                      (CKIN_FILE_NAME,))
    print "\n" + mesg
    return True

def ckin(args):
    check_snew = True
    if len(args) > 0 and args[0] == "--no_check":
        check_snew = False
        args = args[1:]
        
    # Everything else goes into the commit message
    if len(args) == 0:
        print "Please provide a log entry string to ckin"
        return False
    else:
        entry = " ".join(args)

    if check_snew:
        bailout = False
        try:
            bailout = check_up_to_date()
        except BzrToolTransportError, e:
            print e.error_report
            print "It looks like you are having problems getting to the parent for this installation"
            print "(maybe your network connection is down?).  You can re-run with '--no_check' to avoid"
            print "requiring a connection to the parent."
            return True
        except BzrToolError, e:
            print e.error_report
            return True
        if bailout:
            print "Bailing out now"
            return True

    # Establish the top-level directory for this installation and get bzr WorkingTree
    tree, top = get_working_tree(os.getcwd())

    mesg = ""
    # Collect status information, bail if we find conflicts
    status_dict = _collect_tree_status(tree)
    if status_dict['conflicts'] is not None:
        mesg += "Your installation has unresolved conflicts.  Please run 'bzr status' to\n"
        mesg += "identify this problem and then correct it.  You should then rerun prep.\n"
        print mesg
        return True

    checkin_files = None

    # Bazaar doesn't support checking in specific files when there are pending merges,
    # so we only build the checkin_files list if there are no merges
    if status_dict['pending_merges'] is None:
        # Extract list of files to check in from file written in prep step.
        ckin_name = os.path.join(top, CKIN_FILE_NAME)
        locals = dict()
        execfile(ckin_name, dict(), locals)
        if not CKIN_TUPLE_NAME in locals.keys():
            raise BzrToolError("failed to find %s in %s, was this file corrupted?" %
                               (CKIN_TUPLE_NAME, ckin_name))
        checkin_files = list(locals[CKIN_TUPLE_NAME])

    log_file = StringIO()
    try:
        # Establish the top-level directory for this installation and get bzr WorkingTree
        logger = bzrlib.trace.push_log_file(log_file)
        tree, top = get_working_tree(os.getcwd())
        comments_name = os.path.join(top, COMMENT_FILE_NAME)
        changes_name = os.path.join(top, CHANGES_FILE_NAME)

        # Check for existence of comments and change files, insert comments into changes
        if not os.path.isfile(comments_name):
            raise BzrToolError("No file %s found - did you forget to prep?" % (COMMENT_FILE_NAME,))
        if not os.path.isfile(changes_name):
            raise BzrToolError("No file %s found - has your tree been corrupted?" % (CHANGES_FILE_NAME,))

        backup_fd, backup_name = tempfile.mkstemp()
        shutil.copyfile(changes_name, backup_name)
        _update_changes_file(comments_name, changes_name, tree, entry)
        bzrlib.trace.pop_log_file(logger)
    except BzrToolError, e:
        mesg += e.error_report + "\n"
        print mesg
        if log_file.getvalue().split():
            mesg += "ckin also collected this logging information:\n"
            mesg += log_file.getvalue()
            mesg += "\n"
        return
    else:
        mesg += "Added contents of %s to %s\n" % (comments_name, changes_name)

    # We want to make sure changes.txt is always on the list of files to be
    # checked in, so we add it here if we're using this list
    if checkin_files is not None:
        checkin_files.append(CHANGES_FILE_NAME)

    # Do the commit
    try:
        logger = bzrlib.trace.push_log_file(log_file)
        tree.commit(entry, specific_files=checkin_files)
        bzrlib.trace.pop_log_file(logger)
    except bzrlib.errors.BzrError, e:
        mesg = "Error on commit.  Bzr reports:\n %s\n" % (str(e),)
        shutil.move(backup_name, changes_name)
        mesg += "%s restored" % (changes_name,)
        if log_file.getvalue().split():
            mesg += "ckin also collected this logging information:\n"
            mesg += log_file.getvalue()
            mesg += "\n"
    else:
        mesg += "Check-in complete"

    # Make announcements
    print mesg
    return True


def snew(args):
    if len(args) != 0:
        return False
    mesg = ""

    try:
        result = _get_local_and_parent_info()
    except BzrToolTransportError, e:
        print e.error_report
        print "It looks like you are having problems getting to the parent for this installation"
        print "(maybe your network connection is down?)."
        return True
    except BzrToolError, e:
        print e.error_report
        return True

    assert len(result) == 8
    local_tree, local_revno, local_last_revid, parent_tree, parent_revno, parent_last_revid, parent_url, state = result

    if local_revno == parent_revno and local_last_revid == parent_last_revid:
        mesg += "Local revision is %d, which is current for parent: %s\n" % (local_revno, parent_url)        
        mesg += "You are up to date!\n"
    else:
        changes_name = os.path.join(".", CHANGES_FILE_NAME)
        diff_options = '--strip-trailing-cr -C 0'
        out_string_io = StringIO()
        result = bzrlib.diff.show_diff_trees(old_tree=local_tree, new_tree=parent_tree,
                                             to_file=out_string_io, specific_files=(changes_name,),
                                             old_label="local ", new_label="parent ",
                                             external_diff_options=diff_options)
        if result:
            mesg += "Diffing %s with parent version at %s\n" % (changes_name, parent_url)
            mesg += out_string_io.getvalue()
            mesg += "\n\n"
            mesg += "Local revision is %d; parent revision is %s\n" % (local_revno, parent_revno)
        else:
            # This should never happen; if we're not on the same revision as the
            # parent there should certainly be differences in changes.txt.
            mesg += "Internal bzrtool.py error: please report this to Ken Basye\n"
    # Make announcements
    print mesg,
    return True


def usage(progname):
    usage = (("Usage: python %s prep|ckin|snew args\n" % progname) + 
              "(but you probably want to run prep, ckin, or snew instead!)\n" +
              "Command arguments:\n" +
              "  prep [--no_check]\n" +
              "  ckin [--no_check] message\n" +
              "  snew\n")
    print usage
    
COMMAND_MAP = {"prep":prep,
               "ckin":ckin,
               "snew":snew,
               }
    
def main(argv):
    commands = set(("prep", "ckin", "snew"))
    if len(argv) < 2 or argv[1] not in COMMAND_MAP.keys():
        usage(argv[0])
        return
    # run the command; a False return means an argument error
    result = COMMAND_MAP[argv[1]](argv[2:])
    if not result:
        usage(argv[0])
        
if __name__ == '__main__':
    if __file__ == '<stdin>':
        print "TEST MODE FROM EMACS - LINES WITH ***** ARE COMMENTARY ONLY"
        tree, top = get_working_tree(os.getcwd())
        print "***** Current changes.txt header:"
        print _make_changes_header(tree, "Ckin message would be here")

        status_dict = _collect_tree_status(tree)
        print "***** Here's what goes onto the screen:"
        dummy = StringIO()
        _write_comments_file(status_dict, dummy)
        print "***** Here's what goes into comments.txt:"
        print dummy.getvalue()

        print "***** Here's what goes into ckin_files.txt:"
        dummy = StringIO()
        _write_ckin_file(status_dict, dummy)
        print dummy.getvalue()

        print "***** Here's our branch:"
        print tree.branch

        print "***** Here's our branch's parent url:"
        parent_url = tree.branch.get_parent()
        print parent_url

        parent_tree, parent_branch, parent_relpath = bzrdir.BzrDir.open_containing_tree_or_branch(parent_url)
        print "***** Here's our branch's parent branch:"
        print parent_tree, parent_branch, parent_relpath
        
        result = _get_local_and_parent_info()
        if result is None:
            # Error messages will have already been printed
            pass

        assert len(result) == 8
        local_tree, local_revno, local_last_revid, parent_tree, parent_revno, parent_last_revid, parent_url2, state = result
        assert parent_url == parent_url2
        print "***** Here are our local and parent trees:"
        print local_tree, parent_tree
        print "***** Here are our local and parent revnos:"
        print local_revno, parent_revno
        print "***** Here are our local and parent last_revids:"
        print local_last_revid, parent_last_revid
        print "***** Here are our local and parent state:"
        print state

        tree.branch.lock_read()
        
        print "***** Here's our branch's graph:"
        graph = tree.branch.repository.get_graph()
        print graph

        print "***** Here's graph.get_heads([local_last_revid, parent_last_revid]):"
        heads = graph.heads([local_last_revid, parent_last_revid])
        print heads
        print "***** Here's heads == set([local_last_revid])"
        print heads == set([local_last_revid])
        print "***** Here's heads == set([parent_last_revid])"
        print heads == set([parent_last_revid])
        print "***** Here's heads == set([local_last_revid, parent_last_revid])"
        print heads == set([local_last_revid, parent_last_revid])
        
        tree.branch.unlock()


    else:
        main(sys.argv)





