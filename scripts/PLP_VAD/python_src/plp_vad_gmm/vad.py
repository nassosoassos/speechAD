'''
Created on Dec 21, 2011

@author: nassos
'''
import os
import itertools
import math
import struct
import numpy as np
from multiprocessing import Process, cpu_count
import subprocess
import re

WORKING_DIR = os.path.join(os.getcwd(),'tmp')
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

root_dir = '/Users/nassos/Documents/Work/robust_speech_processing/RATS'
file_list = '/Users/nassos/Documents/Work/robust_speech_processing/RATS/lists/training_files.list'
input_dir = os.path.join(root_dir,'sad_ldc2011e86_v2/data')
output_dir = '/Users/nassos/Documents/Work/robust_speech_processing/RATS/features'
plp_file_list = '/Users/nassos/Documents/Work/robust_speech_processing/RATS/lists/plp_training_files.list'
acc_plp_file_list = '/Users/nassos/Documents/Work/robust_speech_processing/RATS/lists/acc_plp_training_files.list'
plp_output_dir = os.path.join(output_dir,'plp')
ldc_annotations_dir = os.path.join(root_dir,'LDC2011E87/LDC2011E87_First_Incremental_SAD_Annotation/data')
ldc_annotations_list = os.path.join(root_dir,'lists','annotations.list')
lab_annotations_dir = os.path.join(root_dir,'lab')
fea_list_abs_paths = os.path.join(root_dir,'lists','fea_abs_paths.list')
ncoefs = 12
n_accum_frames = 31
time_unit = 1e-7

def accumulate_feature_vectors(file_list,n_frames,input_dir,output_dir):
    '''
    Read HTK-formatted feature files and accumulate feature
    vectors from multiple frames so that one can then run
    HLDA or something similar to also model dynamics
    '''
    assert(n_frames%2==1)
    f_list = open(file_list,'r')
    for fname in f_list:
        (pth,fname) = os.path.split(fname.rstrip('\r\n'))
        orig_fname = os.path.join(input_dir,pth,fname)
        target_path = os.path.join(output_dir,pth)
        target_fname = os.path.join(target_path,fname)
        if not os.path.exists(target_path):
            os.makedirs(target_path)

        orig_fea_file = open(orig_fname,"rb")
        try:
            n_samples = struct.unpack('>I',orig_fea_file.read(4))[0]
            sampPeriod = struct.unpack('>I',orig_fea_file.read(4))[0]
            sampSize = struct.unpack('>H',orig_fea_file.read(2))[0]
            parmKind = struct.unpack('>H',orig_fea_file.read(2))[0]
            dt = np.dtype('>f4')
            n_dims = sampSize/4

            '''
            Decided to use numpy at  this point to achieve faster
            read/write times
            '''
            z = np.zeros(n_dims*(n_frames/2))
            x = np.fromfile(orig_fea_file,dtype=dt)
            x = np.hstack((z,x,z))
            x = x.reshape(n_dims,n_samples+n_frames-1,order='F')
            orig_fea_file.close()

            target_fea_file = open(target_fname,'wb')
            new_sampSize = n_frames*sampSize
            header = struct.pack('>IIHH',n_samples,sampPeriod,new_sampSize,parmKind)
            target_fea_file.write(header)
            new_n_dims = n_dims*n_frames
            half_win = n_frames/2

            '''
            This is apparently slow
            '''
            for t in range(half_win,n_samples+half_win):
                s= x[:,t-half_win:t+half_win+1]
                l = list(s.reshape(new_n_dims,1,order='F'))
                line = struct.pack('>'+str(new_n_dims)+'f',*l)
                target_fea_file.write(line)
            target_fea_file.close()

        finally:
            orig_fea_file.close()

def accumulate_feature_vectors_parallel(file_list,n_frames,input_dir,output_dir):
    '''
    Run feature vector accumulation on multiple cores
    '''
    f_list = open(file_list,'r')
    files = f_list.readlines()
    n_files = len(files)

    f_list.close()

    n_processors = cpu_count()
    if n_processors==1:
        n_used_processors = 1
    else:
        n_used_processors = n_processors-1

    n_files_per_processor = int(math.ceil(float(n_files)/float(n_used_processors)))

    file_sets = grouper(n_files_per_processor, files, '')
    procs = []
    for i_proc in range(1,n_used_processors+1):
        p_file_list = os.path.join(WORKING_DIR,'fea'+str(i_proc)+'.list')
        p_file_list_fid = open(p_file_list,'w')
        for f in file_sets.next():
            p_file_list_fid.write(f)
        p_file_list_fid.close()
        argums = (p_file_list, n_frames, input_dir, output_dir, )
        p = Process(target=accumulate_feature_vectors, args=argums)
        p.start()
        procs.append(p)

    for i_proc in range(1, n_used_processors+1):
        p = procs[i_proc-1]
        p.join()

def convert_ldc_annotations_to_lab(in_annotations_list, annotations_dir, lab_dir):
    '''
    Read LDC annotations and convert them to .lab format to be used
    by the HTK tools. The tricky part at this point is that HTK
    wants the lab files to be in a single flat directory structure
    '''
    in_list = open(in_annotations_list,'r')
    lab_names = {}
    n_files = 0
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)

    for fl in in_list:
        fl = fl.rstrip('\r\n')
        n_files = n_files + 1
        fnm = os.path.split(fl)[1]
        bname = os.path.splitext(fnm)[0]
        orig_annotation = os.path.join(annotations_dir,fl)
        o_ann_fid = open(orig_annotation,'r')
        lab_name = bname+'.lab'
        lab_annotation = os.path.join(lab_dir,lab_name)

        ''' Make sure that there are no files with the same name'''
        lab_names[lab_name] = 1
        lab_fid = open(lab_annotation,'w')

        for ln in o_ann_fid:
            fields = re.split('\s+',ln)
            start_time = float(fields[2])/time_unit
            end_time = float(fields[3])/time_unit
            lbl = fields[4]
            lab_fid.write('%i %i %s\n' % (start_time, end_time, lbl))
        lab_fid.close()
        o_ann_fid.close()

    assert(n_files == len(lab_names.keys()))
    in_list.close()

def create_corresponding_annotations_list_assert(in_audio_list, annotations_dir, out_annotations_list):
    '''
    Given a list of audio files, find the corresponding annotations
    '''
    in_list = open(in_audio_list,'r')
    out_list = open(out_annotations_list,'w')
    for fl in in_list:
        fl = fl.rstrip('\r\n')
        (pth,fname) = os.path.split(fl)
        bname = os.path.splitext(fname)[0]
        out_path = re.sub(r'_16000','',pth)
        out_path = re.sub(r'/audio/','/sad/',out_path)
        ann_name = os.path.join(out_path,bname+'.txt')
        if os.path.exists(os.path.join(annotations_dir,ann_name)):
            out_list.write('%s\n' % ann_name)
        else:
            print "Did not find annotation file %s" % os.path.join(annotations_dir,ann_name)

    in_list.close()
    out_list.close()


def create_corresponding_list_assert(in_file_list,out_dir,out_file_list,out_sfx):
    '''
    Change the extension of a filename, check if the new file exists
    and if yes write it in the out_file_list
    '''
    in_list = open(in_file_list,'r')
    out_list = open(out_file_list,'w')

    for fl in in_list:
        fname = os.path.splitext(fl.rstrip('\r\n'))[0]
        out_file = fname+'.'+out_sfx
        if os.path.exists(os.path.join(out_dir,out_file)):
            out_list.write('%s.%s\n' % (fname, out_sfx))

    in_list.close()
    out_list.close()

def grouper(n, iterable, fillvalue=None):
    '''
    grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    '''
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def hcopy_parallel(config_filename,scp_filename):
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
        p_scp_file = os.path.join(WORKING_DIR,'data'+str(i_proc)+'.scp')
        scp_fid = open(p_scp_file,'w')
        for f in file_sets.next():
            scp_fid.write(f)
        scp_fid.close()
        argums = (p_scp_file, config_filename, )
        p = Process(target=hcopy, args=argums)
        p.start()
        procs.append(p)

    for i_proc in range(1, n_used_processors+1):
        p = procs[i_proc-1]
        p.join()

def hcopy(scp_file,config_file):
    '''
    Wrapper for HTK HCopy file
    '''
    cmd=['/usr/local/bin/HCopy','-T','1','-S',scp_file,'-C',config_file]
    out_string = subprocess.check_call(cmd)
    print out_string

def herest_thread(model_dir, data_list, label_dir, model_list, proc_id):
    '''
    Wrapper for HERest in parallel mode
    '''
    assert(proc_id>0)
    cmd = ['HERest','-d',model_dir,'-M', model_dir, '-S',data_list,'-L',label_dir,'-p', str(proc_id), model_list]
    #print cmd
    subprocess.call(cmd)

def herest_join(model_dir, acc_list, label_dir, model_list):
    '''
    Wrapper for HERest (joining functionality in parallel mode)
    '''
    cmd = ['HERest','-d',model_dir,'-M',model_dir,'-S',acc_list,'-L',label_dir,'-p', str(0), model_list]
    #print cmd
    subprocess.call(cmd)

def plp_extract(file_list,n_coefs,input_dir,output_dir):
    '''
    Extract PLP features using HTK.
    It runs on multiple cores if there are multiple available
    '''
    HCopy_config_file = os.path.join(WORKING_DIR,'hcopy_config')
    HCopy_scp_file = os.path.join(WORKING_DIR,'data.scp')
    HTK_config = {}
    HTK_config['TARGETKIND'] = 'PLP_0'
    HTK_config['TARGETRATE'] = 100000
    HTK_config['WINDOWSIZE'] = 250000
    HTK_config['USEHAMMING'] = 'T'
    HTK_config['PREEMCOEF'] = 0.97
    HTK_config['NUMCHANS'] = 24
    HTK_config['CEPLIFTER'] = 22
    HTK_config['NUMCEPS'] = 12
    HTK_config['USEPOWER'] = 'T'
    HTK_config['LPCORDER'] = 12
    HTK_config['SOURCEKIND'] = 'WAVEFORM'
    HTK_config['SOURCEFORMAT'] = 'WAVE'
    HTK_config['SAVEWITHCRC'] = 'F'

    write_htk_config(HTK_config,HCopy_config_file)
    write_scp_file(file_list,input_dir,output_dir,HCopy_scp_file)
    hcopy_parallel(HCopy_config_file,HCopy_scp_file)

def write_htk_config(config_map, filename):
    '''
    Given a dictionary of configuration variables and their properties
    write everything in a proper HTK configuration file
    '''
    config_file = open(filename,'w')

    for prop in config_map.keys():
        config_file.write('%s = %s\n' % (prop,config_map[prop]))

    config_file.close()

def write_file_list_abs_paths_assert(in_file_list, input_dir, out_file_list):
    '''
    Given a list of filenames with relative paths and the corresponding
    root directory write the list of absolute filenames
    '''
    in_list = open(in_file_list,'r')
    out_list = open(out_file_list,'w')
    for fl in in_list:
        fl = fl.rstrip('\r\n')
        file_abs_path = os.path.join(input_dir,fl)
        if os.path.exists(file_abs_path):
            out_list.write('%s\n' % file_abs_path)

    in_list.close()
    out_list.close()

def write_scp_file(file_list,input_dir,output_dir,scp_filename):
    '''
    Given a list of files and an output directory write the corresponding HTK
    scp file for use with HCopy
    '''
    scp_file = open(scp_filename,'w')
    f_list = open(file_list,'r')

    for fname in f_list:
        fname = fname.rstrip('\r\n')
        (path, fl) = os.path.split(fname)
        bname = os.path.splitext(fl)[0]
        orig_fname = os.path.join(input_dir,fname)
        out_path = os.path.join(output_dir,path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        fea_fname = os.path.join(out_path,bname+'.fea')
        scp_file.write('%s %s\n' % (orig_fname,fea_fname))

    f_list.close()
    scp_file.close()

if __name__ == '__main__':
    plp_extract(file_list,ncoefs,input_dir,plp_output_dir)
    create_corresponding_list_assert(file_list,plp_output_dir,plp_file_list,'fea')
    accumulate_feature_vectors(plp_file_list,n_accum_frames,plp_output_dir,output_dir)
    create_corresponding_annotations_list_assert(file_list, ldc_annotations_dir, ldc_annotations_list)
    convert_ldc_annotations_to_lab(ldc_annotations_list, ldc_annotations_dir, lab_annotations_dir)
    write_file_list_abs_paths_assert(plp_file_list,output_dir,fea_list_abs_paths)
