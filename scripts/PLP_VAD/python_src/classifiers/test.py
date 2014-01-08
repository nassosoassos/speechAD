'''
Created on Jan 11, 2012

This module is to test the GMM training and evaluation 
functionality in the classifiers package.

@author: nassos
'''
import argparse
import ConfigParser
import logging
import math
import os

import numpy as np

from classifiers import SailGMM, train_gmm_classifier, test_gmm_classifier, evaluate_gmm_classifier
from htk import htklab, htkmfc
from my_utils import lists

model_definition_string = """
<models>
  <gmm>
    <name> CA </name>
    <mixture>
      <weight> 0.4 </weight>
      <mean> -1 -1 1 1</mean> 
      <covariance type="diag">
        1 2 1 1
      </covariance>
    </mixture>  
    <mixture>
      <weight> 0.6 </weight>
      <mean> -2 -3 1 1</mean> 
      <covariance type="diag">
        1 2 1 1
      </covariance>
    </mixture>
  </gmm>
  <gmm>
    <name> CB </name>
    <mixture>
      <weight> 0.5 </weight>
      <mean> 1 1 1 1</mean> 
      <covariance type="diag">
        1 2 1 1 
      </covariance>
    </mixture>  
    <mixture>
      <weight> 0.5 </weight>
      <mean> 3 4 1 1</mean> 
      <covariance type="diag">
        1 2 1 1
      </covariance>
    </mixture>
  </gmm>
</models>
"""
    
def test_gmm_classification(models=None, n_samples=10000, working_dir=os.getcwd(),
                            apply_hlda=False, hlda_nuisance_dims=2, do_not_generate_features=False):
    test_percentage = 0.2
    samp_period = 0.01
    generate_features = not do_not_generate_features
    
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    fea_dir = os.path.join(working_dir, 'features')
    lab_dir = os.path.join(working_dir, 'lab')
    list_dir = os.path.join(working_dir, 'lists')
    model_dir = os.path.join(working_dir, 'models')
    results_dir = os.path.join(working_dir, 'results')
    if not os.path.exists(list_dir):
        os.makedirs(list_dir)
    if not os.path.exists(fea_dir):
        os.makedirs(fea_dir)
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Create a list of the model names
    model_list = os.path.join(working_dir,'models.list')
    ml = open(model_list,'w')
    labels = []
    for md in models:
        ml.write('{}\n'.format(md.name))
        labels.append(md.name)
    ml.close()
        
    tst_file_list = os.path.join(list_dir,'test_files.list')
    trn_file_list = os.path.join(list_dir,'train_files.list')

    if generate_features:
        fea_file_names = write_feature_files(models, n_samples, 
                                             fea_dir, lab_dir)
        n_fea_files = len(fea_file_names)
        logging.info('Number of feature files: {}'.format(n_fea_files))
        n_train_files = int(round(n_fea_files*(1-test_percentage)))
    
        trn_list = open(trn_file_list,'w')    
        for f_n in fea_file_names[0:n_train_files]:
            trn_list.write("{}\n".format(os.path.join(fea_dir,f_n)))
        trn_list.close()
        tst_list = open(tst_file_list,'w')
        for f_n in fea_file_names[n_train_files:n_fea_files-1]:
            tst_list.write("{}\n".format(os.path.join(fea_dir,f_n)))
        tst_list.close()
       
    n_mixes = models[0].n_components
    n_dims = models[0].means.shape[1]
    train_gmm_classifier.train_M_sized_gmm_classifier(n_mixes, n_dims, 'USER', trn_file_list, 
                                                      model_list, lab_dir, model_dir, apply_hlda, 
                                                      hlda_nuisance_dims)
    model_file = os.path.join(model_dir,'newMacros')
    test_gmm_classifier.test_gmm_classifier(tst_file_list, model_list, model_file, results_dir)

    ref_annotations_list = os.path.join(list_dir,'labels.list')
    hyp_annotations_list = os.path.join(list_dir,'hyp.list')
    lists.create_corresponding_list_assert(tst_file_list,lab_dir,ref_annotations_list,'lab')
    lists.create_corresponding_list_assert(tst_file_list,results_dir,hyp_annotations_list,'rec')
    conf_matrix = evaluate_gmm_classifier.evaluate_results_list(ref_annotations_list, 
                                                                hyp_annotations_list,
                                                                samp_period, labels)
    logging.info('Confusion matrix: {}'.format(conf_matrix))   
    

def write_feature_files(models, n_samples, feature_dir, lab_dir):        
    # Constants that should not make any difference
    samp_period = 0.01
    max_file_samples = 2000
    fea_prefix = 'test'
    htk_time_const = 1e7
    fea_type = htkmfc.USER
        
    model_names = []
    for m in models:
        model_names.append(m.name)
    n_dims = models[0].means.shape[1]
    
    n_fea_files = int(math.ceil(n_samples / float(max_file_samples)))
    samples_estimated = 0
    file_names=[]
    for counter in range(n_fea_files):
        fea_name = fea_prefix+str(counter)
        file_names.append(fea_name+'.fea')
        fea_file_name = os.path.join(feature_dir,fea_name+'.fea')
        lab_file_name = os.path.join(lab_dir,fea_name+'.lab')
        
        n_samples_to_generate = min(max_file_samples, n_samples-samples_estimated)
        samples, ind_vector = SailGMM.sample_gmms(models, n_samples_to_generate)
        samples_estimated += n_samples_to_generate
        
        htk_lab_writer = htklab.HTKlab_write(filename=lab_file_name, 
                                             samp_period=round(samp_period*htk_time_const),
                                             labels=model_names)
        htk_lab_writer.write(ind_vector) 
        htk_fea_writer = htkmfc.HTKFeat_write(filename=fea_file_name,
                                              sampPeriod=int(samp_period*htk_time_const),
                                              veclen=n_dims,
                                              paramKind=fea_type)
        htk_fea_writer.writeall(samples)    
    
    return(file_names)
    

if __name__ == '__main__':
    # Logging configuration
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    conf_parser = argparse.ArgumentParser(add_help = False)
    conf_parser.add_argument('-c',"--conf_file",
                             help="specify configuration file",metavar='FILE')
    c_args, remaining_argv = conf_parser.parse_known_args()
    
    defaults = {
                "working_dir" : "/home/work/speech_activity_detection/experiments/test",
                "model_file" : None,
                "n_samples" : 10000,
                "apply_hlda" : False,
                "hlda_nuisance_dims" : 2, 
                "do_not_generate_features" : False 
                }
       
    parser = argparse.ArgumentParser(
                                     parents = [conf_parser],
                                     description = __doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.set_defaults(**defaults)
    parser.add_argument('--model_file', type=str, help='model definition file')
    parser.add_argument('--working_dir', type=str,
                        help="directory where the experiment's output will lie")
    parser.add_argument('--apply_hlda', default=False, action="store_true",
                        help="flag to determine whether to apply HLDA or not")
    parser.add_argument('--n_samples', type=int, 
                        help="number of samples to be generated from each class")
    parser.add_argument('--hlda_nuisance_dims', type=int, 
                        help="names of the classes for which the classifier is built")
    parser.add_argument('--do_not_generate_features', default=False, action="store_true",
                        help="set this if you want reproducible results and you already have generated feature files")
    
    args = parser.parse_args(remaining_argv)      
    
    if c_args.conf_file:
        config = ConfigParser.SafeConfigParser()
        config.read([c_args.conf_file])
        defaults = dict(config.items("Defaults"))
    
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    
    working_dir = args.working_dir
    model_file = args.model_file
    n_samples = args.n_samples
    
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
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    if model_file != None and os.path.exists(model_file):
        mod_f = open(model_file,'r')
        model_definition_string = mod_f.read()
    
    models = SailGMM.gmm_list_from_string(model_definition_string)
    test_gmm_classification(models=models, n_samples=n_samples, working_dir=args.working_dir, 
                            apply_hlda=args.apply_hlda, hlda_nuisance_dims=args.hlda_nuisance_dims,
                            do_not_generate_features=args.do_not_generate_features)
    
