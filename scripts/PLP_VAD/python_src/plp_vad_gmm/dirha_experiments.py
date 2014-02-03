'''
Created on Dec 29, 2011

Description:
Main script to run DARPA Speech Activity Detection (SAD) experiments.

@author: nassos@SAIL
'''
import argparse
import ConfigParser
import logging
import os
import re
import string
import subprocess
import numpy as np

from my_utils import lists, which
from plp_vad_gmm import vad_gmm_train, vad_gmm, vad_gmm_evaluate, vad_hmm_train

time_unit = 1e-7


def add_quotes(string_var):
    '''
    Description:
    Add quotes to a string. Mainly for use with the matlab_batcher.sh

    Input:
    string_var : string variable, e.g., string_var='test'

    Output:
    out_string_var : out_string_var = "\'test\'"
    '''
    out_string_var = "\'"+string_var+"\'"
    return out_string_var


def convert_ldc_annotations_to_lab(in_annotations_list=None, annotations_dir=None, lab_dir=None, label_map=None):
    '''
    Description:
    Read LDC annotations and convert them to .lab format to be used
    by the HTK tools.
    LDC format example:
    data/dev-1/fsh-alv/audio/A/20706_20110719_044000_10003_fsh-alv_A 0 0.000 0.936 NT automatic alv original
    Corresponding .lab format:
    0 936000 NT

    The tricky part at this point is that HTK
    wants the lab files to be in a single flat directory structure (unless and mlf file is used) and
    so the user needs to take care of possible filename clashes

    Input:
    in_annotations_list : list of LDC annotation files
    annotations_dir : directory where the LDC annotations lie (if this is not given
                      then the paths in the annotations_list are considered to be
                      relative)
    lab_dir : directory where the .lab files will lie (flat directory structure)
    label_map : dictionary file linking labels to models
    '''
    in_list = open(in_annotations_list,'r')
    lab_names = {}
    label_class_mapping = {}
    label_class_mapping_exists = False
    n_files = 0
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)

    if label_map is not None and os.path.exists(label_map):
        l_map = open(label_map,'r')
        for ln in l_map:
            ln = ln.rstrip('\r\n')
            ln_info = ln.split()
            label_class_mapping[ln_info[0]] = ln_info[1]
        l_map.close()
        label_class_mapping_exists = True

    if annotations_dir is None:
        absolute_paths_given = True
    else:
        absolute_paths_given = False

    for fl in in_list:
        fl = fl.rstrip('\r\n')
        n_files = n_files + 1
        fnm = os.path.split(fl)[1]
        bname = os.path.splitext(fnm)[0]
        if absolute_paths_given:
            orig_annotation = fl
        else:
            orig_annotation = os.path.join(annotations_dir,fl)

        o_ann_fid = open(orig_annotation,'r')
        lab_name = bname+'.lab'
        lab_annotation = os.path.join(lab_dir,lab_name)

        ''' Make sure that there are no files with the same name'''
        lab_names[lab_name] = 1
        lab_fid = open(lab_annotation,'w')

        previous_label = None
        previous_start_time = 0
        previous_end_time = 0
        for ln in o_ann_fid:
            fields = re.split('\s+',ln)
            try:
                start_time = float(fields[2])/time_unit
                end_time = float(fields[3])/time_unit
            except:
                print '{} {}'.format(orig_annotation, ln)
                raise
            lbl = fields[4]
            if label_class_mapping_exists:
                try:
                    lbl = label_class_mapping[lbl]
                except:
                    print('{} {}'.format(orig_annotation, lbl))
                    raise
            # There could be consecutive segments with the same label
            # which causes problems to HTK so the subsegments are merged
            if lbl != previous_label and previous_label != None:
                lab_fid.write('%i %i %s\n' % (previous_start_time, previous_end_time, previous_label))
                previous_start_time = start_time
                previous_label = lbl
            else:
                if previous_start_time == 0:
                    # Just in case the first timestamp is greater than zero
                    previous_start_time = start_time
                    previous_label = lbl
            previous_end_time = end_time

        lab_fid.write('%i %i %s\n' % (previous_start_time, previous_end_time, previous_label))
        lab_fid.close()
        o_ann_fid.close()

    assert n_files == len(lab_names.keys()), "n_files: {} lab_names: {}".format(n_files, len(lab_names.keys()))
    in_list.close()


def create_corresponding_annotations_list_assert(in_audio_list, annotations_dir, out_annotations_list):
    '''
    This function is DARPA dataset specific.
    Given a list of audio files (relative paths), write the corresponding annotation
    filenames in a file list. This is essentially to resolve some path naming mismatches
    in the LDC data. The transcription for the audio file:
    sad_ldc2011e86_v2/data/train/fsh-alv/audio/A_16000/20705_20110722_220100_10002_fsh-alv_A.wav
    is in the file:
    LDC2011E87/LDC2011E87_First_Incremental_SAD_Annotation/data/train/fsh-alv/sad/A/20705_20110722_220100_10002_fsh-alv_A.txt
    Input:
    in_audio_list : the list of audio files (relative paths to audio dir)
    annotations_dir : the directory where the annotations lie
    out_annotations_list : the filename of the list of annotation files (relative path to the annotations_dir)
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
            logging.warning("Did not find annotation file {}".format(os.path.join(annotations_dir,ann_name)))

    in_list.close()
    out_list.close()


def test(in_audio_list=None, audio_dir=None, ldc_annotations_list=None, lab_dir=None, results_dir=None, model_list=None,
         model_file=None, feature_type='PLP_0', n_coeffs_per_frame=13, acc_frames=31,
         samp_period=0.01, window_length=0.025, eval_script=None):
    '''
    Test Speech Activity Detection for a list of files given a specific model. Ideally, many
    of the input arguments like samp_period, window_length, n_coeffs_per_frame, acc_frames
    should be read from the model file.

    Input:
    in_audio_list : list of audio files (absolute paths)
    audio_dir : directory where the audio files lie
    ldc_annotations_list : list with the LDC annotation files
    lab_dir : directory where the .lab transcription files lie
    results_dir : directory where the results will be stored
    model_list : file containing a list of the class names
    model_file : an HTK formatted mmf file (containing the gmm models for the different classes
    feature_type : an HTK-formatted string describing the feature_type, e.g., MFCC_0
    n_coeffs_per_frame : number of features per frame
    acc_frames : number of frames to accumulate features over
    samp_period : the frame period (in seconds)
    window_length : the frame duration (in seconds)
    eval_script : the java DARPA evaluation script

    Output:
    conf_matrix : return the confusion matrix
    '''
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run the VAD to get the results
    vad_gmm.vad_gmm_list(in_audio_list, model_list, model_file, feature_type,
                         n_coeffs_per_frame, acc_frames, results_dir, results_dir, samp_period, window_length)

    # The annotations are in .lab format, i.e., start_time end_time label per line
    ref_annotations_list = os.path.join(results_dir,'ref_test_annotations.list')
    hyp_annotations_list = os.path.join(results_dir,'hyp_test_annotations.list')
    lists.create_corresponding_list_assert(in_audio_list, lab_dir, ref_annotations_list,'lab')
    lists.create_corresponding_list_assert(in_audio_list, results_dir, hyp_annotations_list,'rec')

    # Given the results and the reference annotations, evaluate by estimating a confusion matrix
    conf_matrix = vad_gmm_evaluate.vad_gmm_evaluate_frames(ref_annotations_list, hyp_annotations_list,
                                                           samp_period, model_list)
    msg = "{0} \n {1}".format(model_file, conf_matrix)
    logging.info(msg)

    # Estimate accuracy
    n_instances = np.sum(conf_matrix)
    n_correct = np.sum(conf_matrix.diagonal())
    msg = "Accuracy: {0} / {1} = {2}".format(n_correct, n_instances, float(n_correct) / n_instances )
    logging.info(msg)

    if eval_script is not None and os.path.exists(eval_script):
        lists.create_corresponding_list_assert(in_audio_list, lab_dir, ref_annotations_list,'lab.frames.txt')
        lists.create_corresponding_list_assert(in_audio_list, results_dir, hyp_annotations_list,'rec.frames.txt')
        vad_evaluate_darpa(testing_list=in_audio_list, ref_annotations_list=ldc_annotations_list,
                           hyp_annotations_list=hyp_annotations_list, eval_script=eval_script,
                           audio_dir=audio_dir, smp_period=samp_period, window_length=window_length,
                           results_dir=results_dir, task_id='{0}_{1}_{2}'.format(feature_type, str(n_coeffs_per_frame),
                                                                              str(acc_frames)))
    return conf_matrix


def darpa_convert_labels_to_ids_file(filename, class_ids):
    label_file = open(filename,'r')
    lns = label_file.readlines()
    label_file.close()
    label_file = open(filename,'w')
    for ln in lns:
        ln = ln.rstrip('\r\n')
        ln_info = ln.split()
        if ln_info[1] == 'S' or ln_info[1]=='NS' or ln_info[1] == 'NT':
            label_file.write('{0} {1}\n'.format(ln_info[0], class_ids[ln_info[1]]))
        else:
            label_file.write('{0} {1}\n'.format(ln_info[0], ln_info[1]))

    label_file.close()


def darpa_convert_labels_to_ids_file_list(file_list):
    class_ids = {}
    class_ids['S'] = 1
    class_ids['NS'] = 0
    class_ids['NT'] = 0
    r_f_list = open(file_list,'r')
    for fl in r_f_list:
        fl = fl.rstrip('\r\n')
        darpa_convert_labels_to_ids_file(fl, class_ids)
    r_f_list.close()


def vad_evaluate_darpa(testing_list=None, ref_annotations_list=None, hyp_annotations_list=None,
                       eval_script=None, audio_dir=None, smp_period=0.01, window_length=0.02,
                       task_id='SAD', results_dir=None):
    '''
    Run the java DARPA evaluation script by calling the relevant MATLAB function. The
    MATLAB function needs to be in MATLAB path. A shell script (matlab_batcher.sh) that
    also needs to be in path is called to run the matlab command from the shell.

    Input:
    testing_list : list of audio files (absolute paths)
    ref_annotations_list : list of reference annotations (absolute paths)
    hyp_annotations_list : list of hypothesized annotations (absolute paths)
    eval_script : the java evaluation script
    audio_dir : directory where the audio files lie
    smp_period : frame period (in seconds)
    window_length : frame length (in seconds)
    task_id : an id for the task that is evaluated
    results_dir : directory where the results will be stored
    '''
    evaluation_scp_file = os.path.join(results_dir,'evaluation.scp')
    test_files = lists.get_contents(testing_list)
    ref_files = lists.get_contents(ref_annotations_list)
    hyp_files = lists.get_contents(hyp_annotations_list)
    eval_scp = open(evaluation_scp_file,'w')
    for t, r, h in map(None, test_files, ref_files, hyp_files):
        eval_scp.write('{} {} {}\n'.format(t, r, h))
    eval_scp.close()

    darpa_convert_labels_to_ids_file_list(hyp_annotations_list)

    args = '{in_file},{script_path},{audio_dir},{working_dir},{task_id},{smp_period},{window_length}'.format(in_file=add_quotes(evaluation_scp_file),
                                                               script_path=add_quotes(eval_script),
                                                               audio_dir=add_quotes(audio_dir),
                                                               working_dir=add_quotes(results_dir),
                                                               task_id=add_quotes(task_id),
                                                               smp_period=smp_period,
                                                               window_length=window_length)
    cmd = ['matlab_batcher.sh','FindPercentageFromResultFiles',args]
    assert(which.which('matlab_batcher.sh') != None)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

if __name__ == '__main__':
    # Logging configuration
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    conf_parser = argparse.ArgumentParser(add_help = False)
    conf_parser.add_argument('-c',"--conf_file",
                             help="specify configuration file",metavar='FILE')
    c_args, remaining_argv = conf_parser.parse_known_args()

    defaults = {
                "root_dir" : None,
                "working_dir" : "/rmt/work/speech_activity_detection/experiments/dirha_hlda",
                "train_script" : "/rmt/work/speech_activity_detection/lists/training_files.list",
                "test_script" : "/home/work/speech_activity_detection/lists/testing_files.list",
                "audio_dir" : '/home/work/speech_activity_detection/sad_ldc2011e86_v2/data/',
                "feature_type" : 'PLP_0',
                "n_features" : 13,
                "acc_frames" : 31,
                "n_train_iterations" : 10,
                "acc_frame_shift" : 1,
                "sampling_period" : 0.01,
                "window_length" : 0.025,
                "apply_hlda" : 'off',
                "hlda_nuisance_dims" : 358,
                "n_gmm_components" : 4,
                "models" : ['S','NS', 'NT'],
                "evaluation_script" : "/home/work/speech_activity_detection/RES_v1-2/RES_1-2_ScoringEngine.jar",
                "label_map" : "/home/work/speech_activity_detection/scripts/class.map",
                "n_states" : 1
                }

    parser = argparse.ArgumentParser(
                                     parents = [conf_parser],
                                     description = __doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.set_defaults(**defaults)
    parser.add_argument('--root_dir', help='root directory where the SAD setup lies')
    parser.add_argument('--audio_dir', help='directory where the audio files lie')
    parser.add_argument('--train_script', help='list of training audio files, relative to the audio directory')
    parser.add_argument('--test_script', help='list of testing audio files, relative to the audio directory')
    parser.add_argument('--working_dir', help="directory where the experiment's output will lie")
    parser.add_argument('--feature_type', help="feature type string using the HTK conventions, e.g., PLP_0 or MFCC")
    parser.add_argument('--n_features', type=int, help="number of features to be extracted per frame")
    parser.add_argument('--hlda_nuisance_dims', type=int, help="number of dimensions to be discarded using HLDA")
    parser.add_argument('--acc_frames', type=int, help="number of frames to accumulate over")
    parser.add_argument('--acc_frame_shift', type=int, help="period of frame accumulation in frames")
    parser.add_argument('--sampling_period', type=float, help="frame sampling period in seconds")
    parser.add_argument('--window_length', type=float, help="frame length in seconds")
    parser.add_argument('--apply_hlda', help="flag to determine whether to apply HLDA or not")
    parser.add_argument('--models',nargs='+',help="names of the classes for which the classifier is built")
    parser.add_argument('--label_map', help="two-column file with the mapping of labels to model names, dictionary")
    parser.add_argument('--n_gmm_components', nargs='+', type=int, help='number of GMM components to be used')
    parser.add_argument('--evaluation_script', help='darpa script to be used for evaluation (if exists)')
    parser.add_argument('--n_train_iterations', type=int, help='number of training iterations')
    parser.add_argument('--n_states', type=int, help='number of HMM states')

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

        n_gmm_components = defaults['n_gmm_components']
        if (n_gmm_components[0] == "[") and (n_gmm_components[-1] == "]"):
            defaults['n_gmm_components'] = eval(n_gmm_components)
        else:
            defaults['n_gmm_components'] = n_gmm_components

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    root_dir = args.root_dir

    training_scp = args.train_script
    testing_scp = args.test_script
    audio_dir = args.audio_dir
    working_dir = args.working_dir
    feature_type = args.feature_type
    n_coeffs_per_frame = args.n_features
    acc_frames = args.acc_frames
    samp_period = args.sampling_period
    darpa_evaluation_script = args.evaluation_script
    win_length = args.window_length
    if args.apply_hlda=='on':
        apply_hlda = True
    else:
        apply_hlda = False
    hlda_nuisance_dims = args.hlda_nuisance_dims
    models = args.models
    n_gmm_components = args.n_gmm_components
    acc_frame_shift = args.acc_frame_shift
    n_train_iterations = args.n_train_iterations
    label_map = args.label_map
    n_states = args.n_states

    if not os.path.exists(working_dir):
        try:
            os.makedirs(working_dir)
        except:
            logging.exception("Cannot create working directory: {0}".format(working_dir))

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

    # Initializing various filenames
    '''
    Lists of audio files and corresponding transcriptions
    '''
    training_audio_list = os.path.join(working_dir, 'audio_train.list')
    testing_audio_list = os.path.join(working_dir, 'audio_test.list')
    train_lab_annotations_list = os.path.join(working_dir,'train_annotations.list')
    test_lab_annotations_list = os.path.join(working_dir,'test_annotations.list')

    '''
    Directory where the .lab formatted annotations will be stored. These are used by all HTK tools.
    The directory is created if it doesn't exist.
    '''
    lab_annotations_dir = os.path.join(root_dir,'lab')
    model_list = os.path.join(working_dir,'models.list')

    logging.basicConfig(filename=os.path.join(working_dir,'experiment.log'), level=logging.DEBUG)

    '''
    Logging the various configuration parameters.
    '''
    logging.info(vars(args))

    lists.create_lists_from_scp(scp=training_scp,
                                list_of_lists=(training_audio_list, train_lab_annotations_list))
    lists.create_lists_from_scp(scp=testing_scp,
                                list_of_lists=(testing_audio_list, test_lab_annotations_list))

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    m_list = open(model_list,'w')
    m_list.write(string.join(models,'\n'))
    m_list.close()

    # Training the GMM classifier for VAD. Multiple intermediate model files maybe of interest,
    # e.g., with or without HLDA.
    if n_states == 1:
        model_files = vad_gmm_train.vad_gmm_train(audio_list=training_audio_list,
                                                  annotations_dir=lab_annotations_dir,
                                                  model_list=model_list,
                                                  feature_type=feature_type,
                                                  n_coeffs_per_frame=n_coeffs_per_frame,
                                                  acc_frames=acc_frames, acc_frame_shift=acc_frame_shift,
                                                  working_dir=working_dir, apply_hlda=apply_hlda,
                                                  hlda_nuisance_dims=hlda_nuisance_dims,
                                                  n_gmm_comps=n_gmm_components,
                                                  samp_period=samp_period,
                                                  win_length=win_length,
                                                  n_train_iterations = n_train_iterations)
    else:
        model_files = vad_hmm_train.vad_hmm_train(audio_list=training_audio_list,
                                                  annotations_dir=lab_annotations_dir,
                                                  model_list=model_list,
                                                  feature_type=feature_type,
                                                  n_coeffs_per_frame=n_coeffs_per_frame,
                                                  n_states=n_states,
                                                  acc_frames=acc_frames, acc_frame_shift=acc_frame_shift,
                                                  working_dir=working_dir, apply_hlda=apply_hlda,
                                                  hlda_nuisance_dims=hlda_nuisance_dims,
                                                  n_gmm_comps=n_gmm_components,
                                                  samp_period=samp_period,
                                                  win_length=win_length,
                                                  n_train_iterations = n_train_iterations)


    results_dir = os.path.join(working_dir,'results')
    for m in model_files:
        m_result_dir = os.path.join(results_dir,os.path.split(m)[1])
        conf_matrix = test(in_audio_list=testing_audio_list, audio_dir=audio_dir,
                           lab_dir=lab_annotations_dir, results_dir=m_result_dir,
                           model_list=model_list, model_file=m,
                           feature_type=feature_type, n_coeffs_per_frame=n_coeffs_per_frame,
                           acc_frames=acc_frames, samp_period=samp_period, window_length=win_length,
                           eval_script=darpa_evaluation_script)

