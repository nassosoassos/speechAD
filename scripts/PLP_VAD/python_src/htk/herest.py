'''
Description:
Running HTK command HERest on multiple processors. This script
is designed to run transparently.

Created on Dec 27, 2011

@author: Nassos Katsamanis, PhD
'''
from multiprocessing import Process, cpu_count, Queue
import sys
import os
import itertools
import math
import subprocess
import logging
import string
from my_utils.which import which

os.environ['PATH'] += os.pathsep + '/usr/local/bin'
herest_bin = which('HERest')

def grouper(n, iterable, fillvalue=None):
    #grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def herest(args):
    get_scp_file = False
    scp_file_found = False
    already_in_parallel_mode = False
    scp_file = ''
    argums = []
    for arg in args:
        if arg=='-S':
            get_scp_file = True
        elif get_scp_file == True:
            scp_file = arg
            assert(os.path.exists(scp_file))
            scp_file_found = True
            get_scp_file = False
        elif arg=='-p':
            # The calling function is taking care of the parallelization
            already_in_parallel_mode = True
            break
        else:
            argums.append(arg)

    #if scp_file_found and not already_in_parallel_mode:
    #    # In this case use parallel mode
    #    herest_parallel(scp_file, argums)
    #else:
    #    # Run in normal mode
    print args
    herest_serial(args)


def herest_thread(proc_id, scp_file, out_queue, args):
    '''
    This is called by herest_parallel
    '''
    assert(proc_id>0)
    cmd = args
    cmd.insert(0,herest_bin)
    cmd.insert(1,'-S')
    cmd.insert(2,scp_file)
    cmd.insert(1,'-p')
    cmd.insert(2,str(proc_id))
    model_dir_given = False
    model_dir = ''
    for ar in args:
        if ar=='-M':
            model_dir_given = True
        elif model_dir_given:
            model_dir = ar
            assert(os.path.exists(model_dir))
            break
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))
    hacc_file = 'HER'+str(proc_id)+'.acc'
    if model_dir_given:
        hacc_file = os.path.join(model_dir, hacc_file)
    assert(os.path.exists(hacc_file))
    out_queue.put(hacc_file)


def herest_join(acc_list, args):
    cmd = args
    cmd.insert(0,herest_bin)
    cmd.insert(1,'-S')
    cmd.insert(2,acc_list)
    cmd.insert(1,'-p')
    cmd.insert(2,str(0))
    #cmd.insert(1,'-T')
    #cmd.insert(2,'1')
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

def herest_parallel(scp_file, args):
    '''
    Check whether we are on a cluster with a scheduler system installed
    or we are just on a single machine
    '''
    # Read the list of data and split into multiple chunks
    #if which('qsub') == None:
    herest_parallel_multicore(scp_file,args)
    #else:
    #    herest_parallel_cluster(scp_file,args)

def herest_parallel_cluster(scp_file,args):
    pass

def herest_parallel_multicore(scp_file, args):
    '''
    Run HERest in parallel mode on a single machine
    '''
    scp_fid = open(scp_file,'r')
    train_files = scp_fid.readlines()
    n_train_files = len(train_files)
    assert(n_train_files>0)
    scp_fid.close()

    n_processors = cpu_count()
    if n_processors==1:
        n_used_processors = 1
    else:
        n_used_processors = n_processors-1

    n_files_per_processor = int(math.ceil(float(n_train_files)/float(n_used_processors)))
    file_sets = grouper(n_files_per_processor, train_files, '')

    procs = []
    # Queue to hold the .acc file names
    q = Queue()
    for i_proc in range(1,n_used_processors+1):
        l_file = os.path.join('data'+str(i_proc)+'.scp')
        list_fid = open(l_file,'w')
        try:
            for f in file_sets.next():
                list_fid.write(f)
            list_fid.close()
            argums = (i_proc, l_file, q, args )
            p = Process(target=herest_thread, args=argums)
            p.start()
            procs.append(p)
        except StopIteration:
            break

    acc_list = os.path.join('acc.list')
    acc_list_fid = open(acc_list,'w')
    for i_proc in procs:
        i_proc.join()
        her_file = q.get()
        acc_list_fid.write(her_file+'\n')

    acc_list_fid.close()

    herest_join(acc_list, args)

def herest_serial(args):
    cmd = args[:]
    cmd.insert(0,herest_bin)
    #cmd.insert(1,'-T')
    #cmd.insert(2,'1')
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

if __name__ == '__main__':
    nargs = len(sys.argv)
    if nargs == 1:
        cmd = [herest_bin]
        subprocess.call(cmd)
        sys.exit()

    herest(sys.argv[1:])
