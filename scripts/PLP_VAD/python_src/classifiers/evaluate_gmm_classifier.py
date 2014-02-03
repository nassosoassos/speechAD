'''
Created on Dec 29, 2011

@author: nassos
'''
import re
import logging
import os

import numpy

def timestamps_to_frames(frame_period,in_file,out_file):
    '''
    Change the resolution of the annotation from segments
    to frames
    '''
    #print in_file
    in_fid = open(in_file,'r')
    out_fid = open(out_file,'w')

    previous_end_frame = 0
    for line in in_fid:
        m = re.match(r'([\d\.]+)\s+([\d\.]+)\s+([\w]+)',line)
        if m:
            start_time = int(float(m.group(1)))
            end_time = int(float(m.group(2)))
            label = m.group(3)
            start_frame = max(1,int(start_time/frame_period))
            if start_frame<=previous_end_frame:
                start_frame=previous_end_frame+1
            end_frame = max(1,int(end_time/frame_period)-1)
            if end_frame<=start_frame:
                end_frame = start_frame
            previous_end_frame = end_frame
            for fr in range(start_frame,end_frame+1):
                out_fid.write('{0} {1}\n'.format(str(fr),label))

    # Take care of the last frame, which otherwise would be
    # ignored
    out_fid.write('{0} {1}\n'.format(str(end_frame+1),label))

    in_fid.close()
    out_fid.close()

def timestamps_to_frames_list(frame_period,file_list,out_dir,out_sfx):
    '''
    Change the resolution of the annotation for a list of annotation files
    '''
    f_list = open(file_list,'r')
    for ln in f_list:
        fl = ln.rstrip('\r\n')
        fname = os.path.splitext(os.path.split(fl)[1])[0]
        out_fname = os.path.join(out_dir,fname+out_sfx)
        timestamps_to_frames(frame_period, fname, out_fname)

    f_list.close()

def evaluate_results_file(ref_file,hyp_file,samp_period,labels):
    hyp_file_path = os.path.split(hyp_file)[0]
    ref_fname = os.path.split(ref_file)[1]
    frames_ref_file = os.path.join(hyp_file_path, ref_fname + '.frames.txt')
    frames_hyp_file = hyp_file+'.frames.txt'

    samp_period *= 1e7
    timestamps_to_frames(samp_period, ref_file, frames_ref_file)
    timestamps_to_frames(samp_period, hyp_file, frames_hyp_file)

    r_file = open(frames_ref_file,'r')
    hyp_file = open(frames_hyp_file,'r')

    refs = r_file.readlines()
    hyps = hyp_file.readlines()

    r_file.close()
    hyp_file.close()

    n_refs = len(refs)
    n_hyps = len(hyps)

    if n_refs != n_hyps:
      logging.debug('{0}: Unequal number of labels in reference and hypothesis'.format(frames_hyp_file))
    n_labs = min(n_refs,n_hyps)
    n_labels = len(labels)
    label_map = {}
    for count in range(0,n_labels):
        label_map[labels[count]] = count

    conf_matrix = numpy.zeros((n_labels, n_labels), dtype=numpy.int64)

    # In HTK time units
    for count in range(0,n_labs):
        ref_lab = re.split(r'\s+',refs[count])[1]
        hyp_lab = re.split(r'\s+',hyps[count])[1]
        conf_matrix[label_map[ref_lab],label_map[hyp_lab]] += 1

    return conf_matrix

def evaluate_results_list(ref_file_list, hyp_file_list, samp_period, labels, mode='sequence'):
    r_list = open(ref_file_list,'r')
    h_list = open(hyp_file_list,'r')

    ref_files = r_list.readlines()
    hyp_files = h_list.readlines()

    r_list.close()
    h_list.close()

    n_files = len(ref_files)
    assert(n_files == len(hyp_files))
    n_labels = len(labels)
    conf_matrix = numpy.zeros((n_labels, n_labels), dtype=numpy.int64)

    label_map = {}
    for count in range(0,n_labels):
        label_map[labels[count]] = count

    for count in range(0,n_files):
        r_file = ref_files[count].rstrip('\r\n')
        h_file = hyp_files[count].rstrip('\r\n')
        if mode=='sequence':
            conf_matrix += evaluate_results_file(r_file,h_file,samp_period,labels)
        elif mode=='single':
            ref = open(r_file,'r')
            hyp = open(h_file,'r')
            ref_label = ref.readline().rstrip('\r\n')
            hyp_label = hyp.readline().rstrip('\r\n')

            ref_info = re.split(r'\s+',ref_label)
            # Check if start and end times are also given
            if len(ref_info)>2:
                ref_lab = ref_info[2]
            else:
                ref_lab = ref_info[0]
            hyp_lab = re.split(r'\s+',hyp_label)[2]
            conf_matrix[label_map[ref_lab], label_map[hyp_lab]] += 1
            ref.close()
            hyp.close()

    return conf_matrix

