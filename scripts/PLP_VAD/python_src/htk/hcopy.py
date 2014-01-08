'''
Created on Dec 27, 2011

Description:
Running HTK HCopy on multiple processors. The script is
designed to run transparently.

@author: Nassos Katsamanis, PhD
'''
import itertools
import math
from multiprocessing import Process, cpu_count
import subprocess
import sys
import os
import logging
import string
from my_utils.which import which

os.environ['PATH'] += os.pathsep + '/usr/local/bin'
hcopy_bin = which('HCopy')
hcopy_par = which('HCopy.pl')
assert(os.path.exists(hcopy_bin))

def grouper(n, iterable, fillvalue=None):
    #grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def hcopy(args):
    get_scp_file = False
    scp_file_found = False
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
        else:
            argums.append(arg)

    if scp_file_found:
        # In this case use parallel mode
        hcopy_parallel(scp_file, argums)
    else:
        # Run in normal mode
        hcopy_serial(args)

def hcopy_parallel(scp_file, args):
    '''
    Check whether we are on a cluster with a scheduler system installed
    or we are just on a single machine
    '''
    # Read the list of data and split into multiple chunks
    if which('qsub') == None:
        hcopy_parallel_multicore(scp_file,args)
    else:
        hcopy_parallel_cluster(scp_file,args)

def hcopy_parallel_cluster(scp_filename, args):
    args.insert(0,'-S')
    args.insert(1,scp_filename)
    cmd = args
    cmd.insert(0,hcopy_par)
    logging.debug(string.join(cmd,' '))
    subprocess.call(cmd)

def hcopy_parallel_multicore(scp_filename, args):
    '''
    Run HCopy on multiple cores by splitting the data
    into multiple chunks
    '''
    scp_file = open(scp_filename,'r')
    files = scp_file.readlines()
    n_files = len(files)
    scp_file.close()

    n_processors = cpu_count()
    if n_processors==1:
        n_used_processors = 1
    else:
        n_used_processors = n_processors-1
    n_files_per_processor = int(math.ceil(float(n_files)/float(n_used_processors)))

    file_sets = grouper(n_files_per_processor, files, '')
    procs = []
    for i_proc in range(1,n_used_processors+1):
        p_scp_file = os.path.join('data'+str(i_proc)+'.scp')
        scp_fid = open(p_scp_file,'w')
        try:
            for f in file_sets.next():
                scp_fid.write(f)
            scp_fid.close()
            argums = (p_scp_file, args, )
            p = Process(target=hcopy_thread, args=argums)
            p.start()
            procs.append(p)
        except StopIteration:
            break

    for i_proc in procs:
        i_proc.join()

def hcopy_serial(args):
    '''
    Wrapper for HTK HCopy
    '''
    cmd = args
    cmd.insert(0,hcopy_bin)
    logging.debug(string.join(cmd,' '))
    subprocess.call(cmd)

def hcopy_thread(scp_file, args):
    args.insert(0,'-S')
    args.insert(1,scp_file)
    hcopy_serial(args)

if __name__ == '__main__':
    nargs = len(sys.argv)
    if nargs == 1:
        cmd = [hcopy_bin]
        subprocess.call(cmd)
    else:
        hcopy(sys.argv[1:])
