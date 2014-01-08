'''
Created on Jan 28, 2012

Description:
  Train a GMM to perform channel estimation for DARPA evaluation.

@author: nassos@SAIL
'''
import argparse
import ConfigParser
import logging
import os
import string
import wave

from plp_vad_gmm import vad_extract_features as vef
from plp_vad_gmm import vad_gmm_train, vad_gmm_evaluate
from classifiers import test_gmm_classifier
from classifiers import evaluate_gmm_classifier
from my_utils import lists

def scp2lists(scp_file, file_list, lab_annotations_dir):
    scp = open(scp_file,'r')
    f_list = open(file_list,'w')
    if not os.path.exists(lab_annotations_dir):
        os.makedirs(lab_annotations_dir)
    start_time = 0
    for ln in scp:
        ln = ln.rstrip('\r\n')
        ln_info = ln.split()
        f_name = ln_info[0]
        label = ln_info[1]
        f_list.write('{}\n'.format(f_name))
        wv = wave.open(f_name)
        n_samples = wv.getnframes()
        fs = wv.getframerate()
        end_time = int(n_samples *1e7/ fs)
        wv.close()

        lab_file_name = os.path.join(lab_annotations_dir, os.path.splitext(os.path.split(f_name)[1])[0]+'.lab')
        lab_file = open(lab_file_name,'w')
        lab_file.write('{} {} {}\n'.format(start_time, end_time, label))
        lab_file.close()
    scp.close()
    f_list.close()

def estimate_channel_gmm(audio_list, model_list, model_file, feature_type, 
                         n_coeffs_per_frame, acc_frames, results_dir, working_dir):
    features_dir = os.path.join(working_dir,feature_type)
    acc_features_dir = features_dir+'_frames'
    fea_file_list = os.path.join(working_dir,'feature_files.list')
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(acc_features_dir):
        os.makedirs(acc_features_dir)
    vef.fea_extract(audio_list,feature_type,n_coeffs_per_frame,features_dir)
    vef.create_corresponding_list_assert(audio_list,features_dir,fea_file_list,'fea')
    vef.accumulate_feature_vectors(fea_file_list,acc_frames,acc_features_dir)
    vef.create_corresponding_list_assert(audio_list,acc_features_dir,fea_file_list,'fea')
      
    test_gmm_classifier.test_gmm_classifier(test_list=fea_file_list, model_list=model_list, 
                                            model_file=model_file, results_dir=results_dir, 
                                            mode='single')    

def test(in_audio_list=None, lab_dir=None, results_dir=None, model_list=None, 
         model_file=None, feature_type='PLP_0', n_coeffs_per_frame=13, acc_frames=31, 
         samp_period=0.01, window_length=0.02): 
    '''
    Test Speech Activity Detection for a list of files given a specific model. Ideally, many 
    of the input arguments like samp_period, window_length, n_coeffs_per_frame, acc_frames
    should be read from the model file. 
    
    Input:
    in_audio_list : list of audio files (absolute paths)
    lab_dir : directory where the .lab transcription files lie
    results_dir : directory where the results will be stored
    model_list : file containing a list of the class names
    model_file : an HTK formatted mmf file (containing the gmm models for the different classes
    feature_type : an HTK-formatted string describing the feature_type, e.g., MFCC_0
    n_coeffs_per_frame : number of features per frame
    acc_frames : number of frames to accumulate features over
    samp_period : the frame period (in seconds) 
    window_length : the frame duration (in seconds)
    
    Output:
    conf_matrix : return the confusion matrix
    '''
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Classify using 
    estimate_channel_gmm(in_audio_list, model_list, model_file, feature_type, 
            n_coeffs_per_frame, acc_frames, results_dir, results_dir)
    
    # The annotations are in .lab format, i.e., start_time end_time label per line
    ref_annotations_list = os.path.join(results_dir,'ref_test_annotations.list')
    hyp_annotations_list = os.path.join(results_dir,'hyp_test_annotations.list')
    lists.create_corresponding_list_assert(in_audio_list, lab_dir, ref_annotations_list,'lab')
    lists.create_corresponding_list_assert(in_audio_list, results_dir, hyp_annotations_list,'rec')
    
    # Given the results and the reference annotations, evaluate by estimating a confusion matrix
    conf_matrix = vad_gmm_evaluate.vad_gmm_evaluate_frames(ref_annotations_list, hyp_annotations_list, 
                                                           samp_period, model_list, mode='single')
    msg = "{} \n {}".format(model_file, conf_matrix)
    logging.info(msg)
    
    return(conf_matrix)



if __name__ == '__main__':
    # Logging 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    conf_parser = argparse.ArgumentParser(add_help = False)
    conf_parser.add_argument('-c',"--conf_file",
                            help="specify configuration file", metavar='FILE')

    c_args, remaining_argv = conf_parser.parse_known_args()

    defaults = {
                "root_dir" : None,
                "working_dir" : None,
                "train_script" : "training.list",
                "test_script" : "testing.list",
                "feature_type" : "PLP_0",
                "n_features" : 13,
                "acc_frames" : 31,
                "n_train_iterations" : 10,
                "acc_frame_shift" : 1,
                "sampling_period" : 0.01,
                "apply_hlda" : "off",
                "hlda_nuisance_dims" : 364,
                "n_gmm_components" : 4,
                "models" : ['A','B','C','D','E','F','G','H']
               }
   
    parser = argparse.ArgumentParser(
                                    parents = [conf_parser],
                                    description = __doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.set_defaults(**defaults)
    parser.add_argument('--root_dir', help='root directory where the SAD setup lies')
    parser.add_argument('--train_script', help='two-column script of training audio files with their labels')
    parser.add_argument('--test_script', help='two-column script of test audio files with their labels')
    parser.add_argument('--working_dir', help="directory where the experiment's output will lie")
    parser.add_argument('--feature_type', help="feature type string using the HTK conventions, e.g., PLP_0 or MFCC")
    parser.add_argument('--n_features', type=int, help="number of features to be extracted per frame")
    parser.add_argument('--hlda_nuisance_dims', type=int, help="number of dimensions to be discarded using HLDA")
    parser.add_argument('--acc_frames', type=int, help="number of frames to accumulate over")
    parser.add_argument('--acc_frame_shift', type=int, help="period of frame accumulation in frames")
    parser.add_argument('--sampling_period', type=float, help="frame sampling period in milliseconds")
    parser.add_argument('--apply_hlda', help="flag to determine whether to apply HLDA or not")
    parser.add_argument('--models',nargs='+',help="names of the classes for which the classifier is built")
    parser.add_argument('--n_gmm_components', type=int, help='number of GMM components to be used')
    parser.add_argument('--n_train_iterations', type=int, help='number of training iterations')
    
    args = parser.parse_args(remaining_argv)
    root_dir = args.root_dir

    if c_args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([c_args.conf_file])

        if root_dir != None:
            defaults = dict(config.items("Defaults",0,{"root_dir" : root_dir}))
        else:
            defaults = dict(config.items("Defaults",0))
        models = defaults['models']
        if (models[0]== "[") and (models[-1] == "]"):
            defaults['models'] = eval(models)
        else:
            defaults['models'] = models
    
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    root_dir = args.root_dir
    
    training_scp = args.train_script
    testing_scp = args.test_script
    working_dir = args.working_dir
    feature_type = args.feature_type
    n_coeffs_per_frame = args.n_features
    acc_frames = args.acc_frames
    samp_period = args.sampling_period
    if args.apply_hlda=='on':
        apply_hlda = True
    else:
        apply_hlda = False
    hlda_nuisance_dims = args.hlda_nuisance_dims
    models = args.models
    n_gmm_components = args.n_gmm_components
    acc_frame_shift = args.acc_frame_shift
    n_train_iterations = args.n_train_iterations
    
    if not os.path.exists(working_dir):
        try:
            os.makedirs(working_dir)
        except:
            logging.exception("Cannot create working directory: {}".format(working_dir))
    
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(working_dir,'experiment.log'))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    training_audio_list = os.path.join(working_dir, 'audio_train.list')
    testing_audio_list = os.path.join(working_dir, 'audio_test.list')
    lab_annotations_dir = os.path.join(working_dir, 'lab')
    model_list = os.path.join(working_dir, 'models.list')

    scp2lists(training_scp, training_audio_list, lab_annotations_dir)
    scp2lists(testing_scp, testing_audio_list, lab_annotations_dir)

    m_list = open(model_list,'w')
    m_list.write(string.join(models,'\n'))
    m_list.close()


    model_files = vad_gmm_train.vad_gmm_train(audio_list=training_audio_list, 
                                              annotations_dir=lab_annotations_dir, 
                                              model_list=model_list, 
                                              feature_type=feature_type, 
                                              n_coeffs_per_frame=n_coeffs_per_frame, 
                                              acc_frames=acc_frames, acc_frame_shift=acc_frame_shift, 
                                              working_dir=working_dir, apply_hlda=apply_hlda, 
                                              hlda_nuisance_dims=hlda_nuisance_dims, 
                                              n_gmm_comps=n_gmm_components, 
                                              n_train_iterations = n_train_iterations,
                                              samp_period=samp_period)

    results_dir = os.path.join(working_dir,'results')
    for m in model_files:
         m_result_dir = os.path.join(results_dir, os.path.split(m)[1])
         conf_matrix = test(in_audio_list=testing_audio_list, 
                           lab_dir=lab_annotations_dir, results_dir=m_result_dir, 
                           model_list=model_list, model_file=m,
                           feature_type=feature_type, n_coeffs_per_frame=n_coeffs_per_frame, 
                           acc_frames=acc_frames, samp_period=samp_period) 
    

