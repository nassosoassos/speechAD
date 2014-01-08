'''
Created on Dec 28, 2011

@author: nassos
'''
import os
import argparse
import textwrap
import shutil
from classifiers.create_gmm import create_gmm, create_ergodic_hmm
from classifiers import initialize_gmm
from classifiers import train_gmm
from classifiers.train_hlda import train_hlda
from htk.read_htk import read_htk_header

covar_type = 'diag'
max_dimensions = 500

def train_hmm_classifier_incrementally(M, n_states, n_dims, feature_type, training_list, model_list,
                                       lab_dir, model_dir, apply_hlda=False, hlda_nuisance_dims=0,
                                       n_train_iterations=10):
    '''
    Train a HMM-based classifier with a known number of components
    The assumption is that training files are HTK-formatted feature files and that
    there exist HTK-formatted transcription files as well named in the same way
    as the feature files with just a .lab suffix. The lab_dir should correspond to
    a flat structure.
    '''
    # Find the dimensionality of the feature vector
    assert(os.path.exists(training_list))
    t_list = open(training_list,'r')
    fea_file = t_list.readline().rstrip('\r\n')
    t_list.close()
    sampSize = read_htk_header(fea_file)[2]
    orig_n_dims = sampSize/4
    assert(orig_n_dims<max_dimensions)

    # Find the model names
    models = []
    m_list = open(model_list,'r')
    for line in m_list:
        line = line.rstrip('\r\n')
        models.append(line)

    # Directory structure generation
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    hmm0_dir = os.path.join(model_dir,'hmm0')
    if not os.path.exists(hmm0_dir):
        os.makedirs(hmm0_dir)
    hmm1_dir = os.path.join(model_dir,'hmm1')
    if not os.path.exists(hmm1_dir):
        os.makedirs(hmm1_dir)

    n_models = len(models)
    final_model_files = []
    for count in range(n_models):
        if len(M) > 1:
            n_comps = M[count]
        initial_n_comps = 1
        model_name = models[count]

        create_ergodic_hmm(orig_n_dims, n_states, feature_type, model_name, covar_type, hmm0_dir)
        initialize_gmm.initialize_hmm_hinit(model_name, training_list, lab_dir, hmm0_dir, hmm1_dir)

        if count == 0:
            # Estimate variance floors as percentage of the global variances (only once)
            initialize_gmm.estimate_minimum_variances(training_list, os.path.join(hmm0_dir, models[count]), hmm0_dir)
            min_var_macro_file = os.path.join(hmm0_dir,'vFloors')
            train_gmm.prepare_vfloor_macros_file(model_file=os.path.join(hmm1_dir, model_name),
                                                  vfloors_file=min_var_macro_file)


        # Increase the number of mixtures
        n_mixes = 2;
        orig_model_dir = hmm1_dir
        target_model_dir = os.path.join(model_dir,'hmm2');
        while n_mixes < n_comps+1:
            if not os.path.exists(target_model_dir):
                os.makedirs(target_model_dir)
            model_file = os.path.join(orig_model_dir, model_name)
            train_gmm.increase_n_components(n_mixes, model_name, model_file, training_list, lab_dir,
                                            target_model_dir, min_var_macro_file, n_states)
            n_mixes *= 2
            orig_model_dir = target_model_dir
            target_model_dir = os.path.join(model_dir,'hmm{}'.format(str(n_mixes)))

        final_model_files.append(os.path.join(orig_model_dir,model_name))

    # Gather all HMM definitions into a single mmf file
    model_file = os.path.join(model_dir, "newMacros")
    mmf = open(model_file,'w')
    for md in final_model_files:
        md_file = open(md,'r')
        mmf.writelines(md_file.readlines())
        mmf.write('\n')
        md_file.close()
    mmf.close()

    train_gmm.train_gmm_set(model_list, training_list, lab_dir, model_dir, model_file, n_train_iterations, min_var_macro_file)
    model_file = os.path.join(model_dir, "newMacros")
    shutil.copyfile(model_file, model_file+'_no_hlda')

    if apply_hlda:
        if hlda_nuisance_dims>0:
            train_hlda(model_list, hlda_nuisance_dims, training_list, lab_dir, model_dir, min_var_macro_file, n_states)


def train_M_sized_gmm_classifier(M, n_dims, feature_type, training_list, model_list,
                                 lab_dir, model_dir, apply_hlda=False, hlda_nuisance_dims=0,
                                 n_train_iterations=10):
    '''
    Train a GMM-based classifier with a known number of components
    The assumption is that training files are HTK-formatted feature files and that
    there exist HTK-formatted transcription files as well named in the same way
    as the feature files with just a .lab suffix. The lab_dir should correspond to
    a flat structure.
    '''
    # Find the dimensionality of the feature vector
    assert(os.path.exists(training_list))
    t_list = open(training_list,'r')
    fea_file = t_list.readline().rstrip('\r\n')
    t_list.close()
    sampSize = read_htk_header(fea_file)[2]
    orig_n_dims = sampSize/4
    assert(orig_n_dims<max_dimensions)

    # Find the model names
    models = []
    m_list = open(model_list,'r')
    for line in m_list:
        line = line.rstrip('\r\n')
        models.append(line)

    # Directory structure generation
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    hmm0_dir = os.path.join(model_dir,'hmm0')
    if not os.path.exists(hmm0_dir):
        os.makedirs(hmm0_dir)
    hmm1_dir = os.path.join(model_dir,'hmm1')
    if not os.path.exists(hmm1_dir):
        os.makedirs(hmm1_dir)

    n_models = len(models)

    for count in range(n_models):
        if len(M) > 1:
            n_comps = M[count]
        else:
            n_comps = M[0]

        # Define the GMM as a trivial HMM with a single state in HTK format
        create_gmm(orig_n_dims, n_comps, feature_type, model_name, covar_type, hmm0_dir)
        initialize_gmm.initialize_gmm_kmeans(model_name, training_list, lab_dir, hmm0_dir, hmm1_dir)

        if count == 1:
            # Estimate variance floors as percentage of the global variances (only once)
            initialize_gmm.estimate_minimum_variances(training_list, os.path.join(hmm0_dir, models[count]), hmm0_dir)
            min_var_macro_file = os.path.join(model_dir,'vFloors')

    # Gather all HMM definitions into a single mmf file
    model_file = os.path.join(hmm1_dir, "newMacros")
    mmf = open(model_file,'w')
    for md in models:
        md_file = open(os.path.join(hmm1_dir,md),'r')
        mmf.writelines(md_file.readlines())
        mmf.write('\n')
        md_file.close()
    mmf.close()

    train_gmm.train_gmm_set(model_list, training_list, lab_dir, model_dir, model_file, n_train_iterations, min_var_macro_file)
    model_file = os.path.join(model_dir, "newMacros")
    shutil.copyfile(model_file, model_file+'_no_hlda')

    if apply_hlda:
        if hlda_nuisance_dims>0:
            train_hlda(model_list, hlda_nuisance_dims, training_list, lab_dir, model_dir, min_var_macro_file)


def train_M_sized_gmm_classifier_incrementally(M, n_dims, feature_type, training_list, model_list,
                                               lab_dir, model_dir, apply_hlda=False, hlda_nuisance_dims=0,
                                               n_train_iterations=10):
    '''
    Train a GMM-based classifier with a known number of components
    The assumption is that training files are HTK-formatted feature files and that
    there exist HTK-formatted transcription files as well named in the same way
    as the feature files with just a .lab suffix. The lab_dir should correspond to
    a flat structure.
    '''
    # Find the dimensionality of the feature vector
    assert(os.path.exists(training_list))
    t_list = open(training_list,'r')
    fea_file = t_list.readline().rstrip('\r\n')
    t_list.close()
    sampSize = read_htk_header(fea_file)[2]
    orig_n_dims = sampSize/4
    assert(orig_n_dims<max_dimensions)

    # Find the model names
    models = []
    m_list = open(model_list,'r')
    for line in m_list:
        line = line.rstrip('\r\n')
        models.append(line)

    # Directory structure generation
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    hmm0_dir = os.path.join(model_dir,'hmm0')
    if not os.path.exists(hmm0_dir):
        os.makedirs(hmm0_dir)
    hmm1_dir = os.path.join(model_dir,'hmm1')
    if not os.path.exists(hmm1_dir):
        os.makedirs(hmm1_dir)

    n_models = len(models)
    final_model_files = []
    for count in range(n_models):
        if len(M) > 1:
            n_comps = M[count]
        else:
            n_comps = M[0]

        initial_n_comps = 1
        if n_comps>1:
            initial_n_comps = 2
        model_name = models[count]

        # Define the GMM as a trivial HMM with a single state in HTK format
        create_gmm(orig_n_dims, initial_n_comps, feature_type, model_name, covar_type, hmm0_dir)
        initialize_gmm.initialize_gmm_kmeans(model_name, training_list, lab_dir, hmm0_dir, hmm1_dir)

        if count == 0:
            # Estimate variance floors as percentage of the global variances (only once)
            initialize_gmm.estimate_minimum_variances(training_list, os.path.join(hmm0_dir, models[count]), hmm0_dir)
            min_var_macro_file = os.path.join(hmm0_dir,'vFloors')
            train_gmm.prepare_vfloor_macros_file(model_file=os.path.join(hmm1_dir, model_name),
                                                  vfloors_file=min_var_macro_file)


        # Increase the number of mixtures
        n_mixes = 4;
        orig_model_dir = hmm1_dir
        target_model_dir = os.path.join(model_dir,'hmm4');
        while n_mixes < n_comps+1:
            if not os.path.exists(target_model_dir):
                os.makedirs(target_model_dir)
            model_file = os.path.join(orig_model_dir, model_name)
            train_gmm.increase_n_components(n_mixes, model_name, model_file, training_list, lab_dir,
                                            target_model_dir, min_var_macro_file)
            n_mixes *= 2
            orig_model_dir = target_model_dir
            target_model_dir = os.path.join(model_dir,'hmm{0}'.format(str(n_mixes)))

        final_model_files.append(os.path.join(orig_model_dir,model_name))

    # Gather all HMM definitions into a single mmf file
    model_file = os.path.join(model_dir, "newMacros")
    mmf = open(model_file,'w')
    for md in final_model_files:
        md_file = open(md,'r')
        mmf.writelines(md_file.readlines())
        mmf.write('\n')
        md_file.close()
    mmf.close()

    train_gmm.train_gmm_set(model_list, training_list, lab_dir, model_dir, model_file, n_train_iterations, min_var_macro_file)
    model_file = os.path.join(model_dir, "newMacros")
    shutil.copyfile(model_file, model_file+'_no_hlda')

    if apply_hlda:
        if hlda_nuisance_dims>0:
            train_hlda(model_list, hlda_nuisance_dims, training_list, lab_dir, model_dir, min_var_macro_file)

def train_gmm_classifier(n_dims, feature_type, training_list, model_list, lab_dir, model_dir, apply_hlda=False, hlda_nuisance_dims=0):
    '''
    Ideally this would automatically determine the size of the GMM
    via cross validation on the training set. Currently, the number of components
    is fixed.
    '''
    M = 4
    train_M_sized_gmm_classifier(M, n_dims, feature_type, training_list, model_list, lab_dir, model_dir, apply_hlda, hlda_nuisance_dims)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GMM classifier using HTK.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent('''\
                                     Example usage:
                                         train_gmm_classifier.py 16 PLP_0 data.scp model.list labs models
                                     '''))
    parser.add_argument('n_comps', metavar='n_comps',
                        type=int, help='number of GMM components')
    parser.add_argumens('n_dims', metavar='n_dims',
                        type=int, help='number of feature dimensions on which the model will be built')
    parser.add_argument('feature_type', metavar='feature_type',
                        type=str, help='HTK-identifier of the feature type')
    parser.add_argument('data_list', metavar='data_list',
                        type=str, help='list of the feature files (absolute paths)' )
    parser.add_argument('model_list',metavar='model_list',
                        type=str, help='list of class names/models')
    parser.add_argument('label_dir', metavar='label_dir',
                        type=str, help='directory where the label files lie')
    parser.add_argument('model_dir', metavar='model_dir',
                        type=str, help='directory where the classifier models will be stored')
    args = parser.parse_args()

    train_M_sized_gmm_classifier(args.n_comps, args.feature_type, args.data_list, args.model_list, args.label_dir, args.model_dir)
