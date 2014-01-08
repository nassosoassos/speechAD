'''
Created on Feb 3, 2012

Description:
  Given a list of audio files identify their noisy channel.

@author: nassos@SAIL
'''
import argparse
import logging
import os
import string

import channel_estimation_experiments

def collect_results(out_file, in_list, rec_dir):
    o_list = open(out_list,'w')
    test = open(test_list,'r')
    for ln in test:
        ln = ln.rstrip('\r\n')
        b_name = os.path.splitext(os.path.split(ln)[1])[0]
        r_file = open(os.path.join(rec_dir,b_name+'.rec'))
        r_file_ln = r_file.readline()
        lab_info = r_file_ln.split()
        if len(lab_info)>1:
            lab = lab_info[2]
        else:
            lab = lab_info[0]
        r_file.close()
        o_list.write('{} {}\n'.format(ln,lab))

    o_list.close()
    test.close()
    

if __name__ == '__main__':
    # Logging 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    defaults = {
                "working_dir" : None,
                "in_list" : None,
                "out_list" : "out.list",
                "test_list" : "testing.list",
                "feature_type" : "PLP_0",
                "n_features" : 13,
                "acc_frames" : 1,
                "sampling_period" : 0.01,
                "model_file" : None,
                "models" : ['A','B','C','D','E','F','G','H']
               }
 

    parser = argparse.ArgumentParser(
                                    description = __doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.set_defaults(**defaults)
    parser.add_argument('--in_list', help='test audio file list')
    parser.add_argument('--out_list', help='list of results')
    parser.add_argument('--working_dir', help="directory where the experiment's output will lie")
    parser.add_argument('--feature_type', help="feature type string using the HTK conventions, e.g., PLP_0 or MFCC")
    parser.add_argument('--n_features', type=int, help="number of features to be extracted per frame")
    parser.add_argument('--acc_frames', type=int, help="number of frames to accumulate over")
    parser.add_argument('--sampling_period', type=float, help="frame sampling period in milliseconds")
    parser.add_argument('--model_file', help='htk-formatted mmf file where the models are stored')
    parser.add_argument('--models', help='list of models')

    args = parser.parse_args()
    working_dir = args.working_dir
    test_list = args.in_list
    out_list = args.out_list
    feature_type = args.feature_type
    n_coeffs_per_frame = args.n_features
    acc_frames = args.acc_frames
    samp_period = args.sampling_period
    models = args.models
    model_file = args.model_file

    if not os.path.exists(working_dir):
        try:
            os.makedirs(working_dir)
        except:
            logging.exception("Cannot create working directory: {}".format(working_dir))
    
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(ch)

    model_list = os.path.join(working_dir, 'models.list')

    m_list = open(model_list,'w')
    m_list.write(string.join(models,'\n'))
    m_list.close()

    results_dir = os.path.join(working_dir,'results')
    #channel_estimation_experiments.estimate_channel_gmm(test_list, model_list, model_file,
    #                                                    feature_type, n_coeffs_per_frame,
    #                                                    acc_frames, results_dir, working_dir)

    collect_results(out_list, test_list, results_dir)
