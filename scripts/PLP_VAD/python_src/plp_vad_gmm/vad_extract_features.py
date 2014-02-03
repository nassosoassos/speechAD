'''
Created on Dec 29, 2011

@author: nassos
'''
import numpy as np
import struct
import os
from multiprocessing import Process, cpu_count
import math
import itertools
import shutil
import re
import logging

from htk import htkmfc
from htk.hcopy import hcopy

def accumulate_feature_vectors(file_list,n_frames,output_dir,acc_frame_shift=1):
    '''
    Read HTK-formatted feature files and accumulate feature
    vectors from multiple frames so that one can then run
    HLDA or something similar to also model dynamics
    '''
    assert(n_frames%2==1)
    f_list = open(file_list,'r')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in f_list:
        fname = fname.rstrip('\r\n')
        target_fname = os.path.join(output_dir,os.path.split(fname)[1])
        if n_frames>1:
            orig_fea_file = open(fname,"rb")
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
                half_win = n_frames/2
                new_range = range(half_win, n_samples + half_win, acc_frame_shift)
                new_n_samples = len(new_range)
                new_sampSize = n_frames*sampSize
                new_sampPeriod = acc_frame_shift*sampPeriod
                header = struct.pack('>IIHH', new_n_samples, new_sampPeriod, new_sampSize, parmKind)
                target_fea_file.write(header)
                new_n_dims = n_dims*n_frames

                '''
                This is apparently slow
                '''
                for t in new_range:
                    s= x[:,t-half_win:t+half_win+1]
                    l = list(s.reshape(new_n_dims,1,order='F'))
                    line = struct.pack('>'+str(new_n_dims)+'f',*l)
                    target_fea_file.write(line)
                target_fea_file.close()

            finally:
                orig_fea_file.close()
        else:
            shutil.copyfile(fname, target_fname)


def accumulate_feature_vectors_parallel(file_list,n_frames,output_dir,acc_frame_shift=1):
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
        p_file_list = os.path.join(output_dir,'fea'+str(i_proc)+'.list')
        p_file_list_fid = open(p_file_list,'w')
        try:
            c_set = file_sets.next()
            if c_set is None:
                break
            for f in c_set:
                p_file_list_fid.write(f)
            p_file_list_fid.close()
            argums = (p_file_list, n_frames, output_dir, acc_frame_shift)
            p = Process(target=accumulate_feature_vectors, args=argums)
            p.start()
            procs.append(p)
        except StopIteration:
            break

    for i_proc in procs:
        p.join()


def create_corresponding_list_assert(in_file_list,out_dir,out_file_list,out_sfx):
    '''
    Change the extension of a filename, check if the new file exists
    and if yes write it in the out_file_list
    '''
    in_list = open(in_file_list,'r')
    out_list = open(out_file_list,'w')

    for fl in in_list:
        fname = os.path.splitext(os.path.split(fl.rstrip('\r\n'))[1])[0]
        out_file = os.path.join(out_dir,fname+'.'+out_sfx)
        if os.path.exists(out_file):
            out_list.write(out_file+'\n')

    in_list.close()
    out_list.close()


def fea_extract(file_list,fea_type,n_coefs,output_dir,samp_period=0.01,win_length=0.025):
    '''
    Extract PLP features using HTK.
    It runs on multiple cores if there are multiple available
    '''
    HCopy_config_file = os.path.join(output_dir,'hcopy_config')
    HCopy_scp_file = os.path.join(output_dir,'data.scp')
    HTK_config = {}
    HTK_config['TARGETKIND'] = fea_type
    HTK_config['TARGETRATE'] = int(1e7*samp_period)
    HTK_config['WINDOWSIZE'] = int(1e7*win_length)
    HTK_config['USEHAMMING'] = 'T'
    HTK_config['PREEMCOEF'] = 0.97
    HTK_config['NUMCHANS'] = 32
    HTK_config['CEPLIFTER'] = 22
    HTK_config['ENORMALISE'] = 'F'
    if re.search('_A',fea_type) and re.search('_D',fea_type):
        num_ceps = n_coefs / 3
    elif re.search('_D', fea_type):
        num_ceps = n_coefs / 2
    else:
        num_ceps = n_coefs
    if re.search('_0',fea_type):
        num_ceps = num_ceps - 1

    HTK_config['NUMCEPS'] = num_ceps
    HTK_config['USEPOWER'] = 'F'
    HTK_config['LPCORDER'] = 12
    HTK_config['SOURCEKIND'] = 'WAVEFORM'
    HTK_config['SOURCEFORMAT'] = 'WAVE'
    HTK_config['SAVEWITHCRC'] = 'F'

    write_htk_config(HTK_config,HCopy_config_file)
    write_scp_file(file_list,output_dir,HCopy_scp_file)
    hcopy_args = ['-C',HCopy_config_file,'-S',HCopy_scp_file]
    hcopy(hcopy_args)

def grouper(n, iterable, fillvalue=None):
    '''
    grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    '''
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def split_feature_files_labels(fea_file_list, annotations_dir, label_list, boundaries=[1/3.0, 2/3.0]):
    '''
    Split the feature files into smaller single-label files
    '''
    f_list = open(fea_file_list,'r')
    files = f_list.readlines()
    f_list.close()

    lbl_list = open(label_list,'r')
    labels = []
    for lbl in lbl_list:
        lbl = lbl.rstrip('\r\n')
        labels.append(lbl)
    lbl_list.close()

    f_list = open(fea_file_list,'w')
    for fl in files:
        fl = fl.rstrip('\r\n')
        feat_dir = os.path.split(fl)[0]
        fl_bname = os.path.splitext(os.path.split(fl)[1])[0]
        htk_reader = htkmfc.HTKFeat_read(fl)
        samp_period = htk_reader.sampPeriod
        vec_len = htk_reader.sampSize/4
        parmkind = htk_reader.parmKind
        data = htk_reader.getall()
        lab_file = os.path.join(annotations_dir, fl_bname+'.lab')
        trans = open(lab_file,'r')
        seg_id = 0
        for trans_ln in trans:
            trans_ln = trans_ln.rstrip('\r\n')
            trans_info = trans_ln.split()
            assert(len(trans_info)>2)
            start_frame = int(trans_info[0])/samp_period
            end_frame = int(trans_info[1])/samp_period
            duration = end_frame - start_frame
            start_frame += int(boundaries[0]*duration)
            end_frame = start_frame + int(boundaries[1]*duration)
            n_samples = start_frame - end_frame + 1
            lbl = trans_info[2]
            if lbl in labels:
                new_fea_segment_bname = fl_bname + '_{0}'.format(str(seg_id))
                seg_id += 1
                new_fea_file = os.path.join(feat_dir, new_fea_segment_bname+'.fea')
                f_list.write('{0}\n'.format(new_fea_file))
                new_lab_file = os.path.join(annotations_dir, new_fea_segment_bname+'.lab')
                new_lab = open(new_lab_file,'w')
                new_lab.write('{0}\n'.format(lbl))
                new_lab.close()
                htk_writer = htkmfc.HTKFeat_write(filename=new_fea_file,
                                                  sampPeriod=samp_period, paramKind=parmkind,
                                                  veclen=vec_len)
                htk_writer.writeall(data[start_frame:end_frame,:])
                htk_writer.close()
        trans.close()
    f_list.close()

def write_htk_config(config_map, filename):
    '''
    Given a dictionary of configuration variables and their properties
    write everything in a proper HTK configuration file
    '''
    config_file = open(filename,'w')

    for prop in config_map.keys():
        config_file.write('%s = %s\n' % (prop,config_map[prop]))

    config_file.close()

def write_scp_file(file_list,output_dir,scp_filename):
    '''
    Given a list of files and an output directory write the corresponding HTK
    scp file for use with HCopy
    '''
    scp_file = open(scp_filename,'w')
    f_list = open(file_list,'r')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in f_list:
        fname = fname.rstrip('\r\n')
        fl = os.path.split(fname)[1]
        bname = os.path.splitext(fl)[0]
        fea_fname = os.path.join(output_dir,bname+'.fea')
        scp_file.write('{0} {1}\n'.format(fname,fea_fname))

    f_list.close()
    scp_file.close()

