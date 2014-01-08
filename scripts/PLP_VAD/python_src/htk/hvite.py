'''
Created on Dec 27, 2011

Description:
Running HTK HVite on multiple processors. The script is
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
hvite_bin = which('HVite')
hvite_par = which('HVite.pl')

def concatenate_mlfs(out_mlf, mlfs):
    mlf_file = open(out_mlf,'w')
    mlf_file.write('#!MLF!#')
    for fl in mlfs:
        if os.path.exists(fl):
            fl_fid = open(fl,'r')
            # Ignore the first line
            fl_fid.readline()
            for line in fl_fid:
                mlf_file.write(line)

    mlf_file.close()


def grouper(n, iterable, fillvalue=None):
    #grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def hvite(args):
    get_scp_file = False
    get_mlf_file = False
    scp_file_found = False
    scp_file = None
    mlf_file = None
    argums = []
    for arg in args:
        if arg=='-S':
            get_scp_file = True
        elif get_scp_file == True:
            scp_file = arg
            assert(os.path.exists(scp_file))
            scp_file_found = True
            get_scp_file = False
        elif arg=='-i':
            get_mlf_file = True
        elif get_mlf_file == True:
            mlf_file = arg
            get_mlf_file = False
        else:
            argums.append(arg)

    if scp_file_found:
        # In this case use parallel mode
        hvite_parallel(scp_file, mlf_file, argums)
    else:
        # Run in normal mode
        hvite_serial(args)

def hvite_parallel(scp_file, mlf_file, args):
    '''
    Check whether we are on a cluster with a scheduler system installed
    or we are just on a single machine
    '''
    # Read the list of data and split into multiple chunks
    #if which('qsub') == None:
    hvite_parallel_multicore(scp_file, mlf_file, args)
    #else:
    # hvite_parallel_cluster(scp_file, mlf_file, args)

def hvite_parallel_cluster(scp_filename, mlf_filename, args):
    args.insert(0,'-S')
    args.insert(1,scp_filename)
    if mlf_filename != None:
        args.insert(0,'-i')
        args.insert(1,mlf_filename)
    cmd = args
    cmd.insert(0,hvite_par)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

def hvite_parallel_multicore(scp_filename, mlf_filename, args):
    '''
    Run HVite on multiple cores by splitting the data
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
    mlfs = []
    for i_proc in range(1,n_used_processors+1):
        p_scp_file = 'data'+str(i_proc)+'.scp'

        scp_fid = open(p_scp_file,'w')
        try:
            for f in file_sets.next():
                scp_fid.write(f)
            scp_fid.close()
            if mlf_filename != None:
                p_mlf_file = 'out'+str(i_proc)+'.mlf'
                mlfs.append(p_mlf_file)
            else:
                p_mlf_file = None

            argums = (p_scp_file, p_mlf_file, args, )
            p = Process(target=hvite_thread, args=argums)
            p.start()
            procs.append(p)
        except StopIteration:
            break

    proc_no = 1
    for i_proc in procs:
        i_proc.join()
        p_scp_file = 'data'+str(proc_no)+'.scp'
        os.remove(p_scp_file)
        proc_no += 1


    if len(mlfs)>0:
        concatenate_mlfs(mlf_filename,mlfs)


def hvite_serial(args):
    '''
    Wrapper for HTK HCopy
    '''
    cmd = args
    cmd.insert(0,hvite_bin)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

def hvite_thread(scp_file, mlf_file, args):
    args.insert(0,'-S')
    args.insert(1,scp_file)
    if mlf_file != None:
        args.insert(0,'-i')
        args.insert(1,mlf_file)
    hvite_serial(args)

if __name__ == '__main__':
    nargs = len(sys.argv)
    if nargs == 1:
        cmd = [hvite_bin]
        subprocess.call(cmd)
    else:
        hvite(sys.argv[1:])
