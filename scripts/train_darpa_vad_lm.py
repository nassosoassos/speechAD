'''
Created on February 4th, 2012

Description:
  Train a voice activity detection Language Model.

nassos@SAIL
'''
import argparse
import logging
import string
import subprocess
import os

from my_utils import which

os.environ['PATH'] += os.pathsep + '/usr/local/bin'
ngram_bin = which.which('ngram-count')

def ngram_count(args):
    cmd = args
    cmd.insert(0, ngram_bin)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_output(cmd, stderr=subprocess.STDOUT))

def ldc_annotations_to_symbol_sequences(label_map, ldc_annotations_list, samp_period, out_file):
    '''
    Convert LDC annotations to sequences of symbols that can be given to SRILM to estimate the 
    n-gram models
    '''
    ldc_list = open(ldc_annotations_list,'r')

    out = open(out_file,'w')
    for fl in ldc_list:
        fl = fl.rstrip('\r\n')
        annot = open(fl, 'r')
        for ln in annot:
            ln = ln.rstrip('\r\n')
            ln_info = ln.split()
            start_time = ln_info[2]
            end_time = ln_info[3]
            label = ln_info[4]
            duration = float(end_time) - float(start_time)
            n_frames = int(duration / float(samp_period))
            if n_frames > 60000:
                print ln
            for _ in range(n_frames):
                out.write('{} '.format(label_map[label]))
        out.write('\n')
        annot.close()
    out.close()
    ldc_list.close()

def train_ngram(src_file, ngram_file, ngram_order):
    args = ['-order', str(ngram_order), '-wbdiscount', '-lm', ngram_file,
            '-text', src_file]
    ngram_count(args)

    
if __name__ == '__main__':
    label_map = {
        'S' : 0,
        'NS': 1,
        'NT': 2,
        'RX': 3
        }

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,)

    parser.add_argument('--order', type=int, help='ngram order')
    parser.add_argument('--ngram_file', help='the file where the ngram will be stored')
    parser.add_argument('--sampling_period', type=float, help='frame sampling period in seconds')
    parser.add_argument('--annotations', help='list of LDC annotation files')
    parser.add_argument('--seq_file', help='file where the symbol sequences will be stored')
    args = parser.parse_args()

    ngram_order = args.order
    ngram_file = args.ngram_file
    samp_period = args.sampling_period
    annotations_list = args.annotations
    seq_file = args.seq_file
   
    ldc_annotations_to_symbol_sequences(label_map, annotations_list, samp_period, seq_file)
    train_ngram(seq_file, ngram_file, ngram_order)

