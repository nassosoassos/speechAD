'''
Created on Dec 28, 2011

@author: nassos
'''
import os
import argparse
import subprocess
import textwrap
import logging
import string
from my_utils.which import which

os.environ['PATH'] += os.pathsep + '/usr/local/bin'

def add_slashes(string_var):
    string_var = "\'"+string_var+"\'"
    return string_var

def create_gmm_matlab(n_dims, n_comps, feat_type, model_name, covar_type='diag', model_dir=None):
    if model_dir == None:
        model_dir =  os.getcwd()
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    print add_slashes(feat_type);
    args = '{dims},{comps},{covar},{feat},{name},{dir}'.format(dims=n_dims,comps=n_comps,
                                                               feat=add_slashes(feat_type),
                                                               covar=add_slashes(covar_type),
                                                               name=add_slashes(model_name),
                                                               dir=add_slashes(model_dir))
    cmd = ['matlab_batcher.sh','define_HTK_GMM',args]
    assert(which('matlab_batcher.sh') != None)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

def create_ergodic_hmm_matlab(n_dims, n_states, feat_type, model_name, covar_type='diag', model_dir=None):
    if model_dir == None:
        model_dir =  os.getcwd()
    else:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    args = '{dims},{n_states},{feat},{name},{dir}'.format(dims=n_dims,n_states=n_states,
                                                          feat=add_slashes(feat_type),
                                                          name=add_slashes(model_name),
                                                          dir=add_slashes(model_dir))
    cmd = ['matlab_batcher.sh','define_HTK_HMM',args]
    assert(which('matlab_batcher.sh') != None)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))


def create_gmm(n_dims, n_comps, feat_type, model_name, covar_type='diag', model_dir=None):
    create_gmm_matlab(n_dims, n_comps, feat_type, model_name, covar_type, model_dir)

def create_ergodic_hmm(n_dims, n_states, feat_type, model_name, covar_type='diag', model_dir=None):
    create_ergodic_hmm_matlab(n_dims, n_states, feat_type, model_name, covar_type, model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a prototype GMM.',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent('''\
                                     Example usage:
                                         create_gmm.py 13 3 PLP_0 NS diag
                                     '''))
    parser.add_argument('n_dims', metavar='n_dims',
                        type=int, help='feature vector dimension' )
    parser.add_argument('n_comps', metavar='n_comps',
                        type=int, help='number of Gaussian components')
    parser.add_argument('feat_type', metavar='feat_type',
                        type=str, help='HTK identifier for the feature type')
    parser.add_argument('model_name', metavar='model_name',
                        type=str, help='name of the GMM to be generated')
    parser.add_argument('covar_type', metavar='covar_type',
                        type=str, help='type of the covariance matrix: diag or full')
    parser.add_argument('--model_dir', metavar='model_dir',
                        type=str, default=None, help='directory where the models will be stored')
    args = parser.parse_args()

    create_gmm(args.n_dims, args.n_comps, args.feat_type, args.model_name, args.covar_type, args.model_dir)
