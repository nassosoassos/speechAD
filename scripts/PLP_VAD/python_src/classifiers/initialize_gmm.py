'''
Created on Dec 28, 2011

@author: nassos
'''
import os
import subprocess
import argparse
import textwrap
import logging
import string
from my_utils import which
os.environ['PATH'] += os.pathsep + '/usr/local/bin'
hinit_bin = which.which('HInit')
hcompv_bin = which.which('HCompV')

n_iterations = 10

def add_quotes(string_var):
    string_var = "\'"+string_var+"\'"
    return string_var

def estimate_minimum_variances(feature_file_list, model_file, out_dir, min_var_factor=0.01):
    '''
    Use HCompV to generate the vFloors file that contains the minimum variances
    allowed for the models to be trained
    '''
    h_cfg_file = os.path.join(out_dir,'hcompv.cfg')
    h_cfg = open(h_cfg_file,'w')
    h_cfg.write('MINVARFLOOR = 0.001')
    h_cfg.close()
    hcompv_args = ['-C', h_cfg_file, '-f', str(min_var_factor), '-m', '-S', feature_file_list,
                   '-M', out_dir, '-o','proto', model_file];
    hcompv(hcompv_args)

def hcompv(args):
    cmd = args
    cmd.insert(0, hcompv_bin)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

def hinit(args):
    cmd = args
    cmd.insert(0, hinit_bin)
    logging.debug(string.join(cmd,' '))
    subprocess.call(cmd)

def initialize_gmm_hinit(model_name, data_list, label_dir, in_model_dir, out_model_dir=None ):
    if out_model_dir==None:
        out_model_dir = in_model_dir
    else:
        if not os.path.exists(out_model_dir):
            os.makedirs(out_model_dir)
    config_file = os.path.join(out_model_dir,'hinit.config')
    conf = open(config_file,'w')
    conf.write('KEEPDISTINCT=TRUE\n')
    conf.close()
    hinit_args = ['-T',str(1),'-S',data_list,'-i',str(n_iterations),'-m',str(1),'-M',out_model_dir,
                  '-C',config_file,'-l',model_name,'-H',os.path.join(in_model_dir,model_name),
                  '-L',label_dir,model_name]
    print(hinit_args)
    hinit(hinit_args)

def initialize_hmm_hinit(model_name, data_list, label_dir, in_model_dir, out_model_dir=None ):
    if out_model_dir==None:
        out_model_dir = in_model_dir
    else:
        if not os.path.exists(out_model_dir):
            os.makedirs(out_model_dir)
    config_file = os.path.join(out_model_dir,'hinit.config')
    conf = open(config_file,'w')
    conf.write('KEEPDISTINCT=TRUE\n')
    conf.close()
    hinit_args = ['-T',str(1),'-S',data_list,'-i',str(n_iterations),'-m',str(1),'-M',out_model_dir,
                  '-C',config_file,'-l',model_name,'-H',os.path.join(in_model_dir,model_name),
                  '-L',label_dir,model_name]
    hinit(hinit_args)

def initialize_gmm_kmeans(model_name, data_list, label_dir, in_model_dir, out_model_dir=None ):
    if out_model_dir==None:
        out_model_dir = in_model_dir
    else:
        if not os.path.exists(out_model_dir):
            os.makedirs(out_model_dir)

    model_file = os.path.join(in_model_dir, model_name)
    args = '{model_file},{data_list},{label_dir},{out_model_dir}'.format(model_file=add_quotes(model_file),
                                                                         data_list=add_quotes(data_list),
                                                                         label_dir=add_quotes(label_dir),
                                                                         out_model_dir=add_quotes(out_model_dir))
    cmd = ['matlab_batcher.sh','initialize_HTK_GMM',args]
    assert(which.which('matlab_batcher.sh') != None)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize a GMM using K-means (as implemented in HTK).',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent('''\
                                     Example usage:
                                         initialize_gmm.py S data.scp label_dir hmm0
                                     '''))
    parser.add_argument('model_name', metavar='model_name',
                        type=str, help='name of the GMM to be initialized')
    parser.add_argument('data_list', metavar='data_list',
                        type=str, help='list of the feature files (absolute paths)' )
    parser.add_argument('label_dir', metavar='label_dir',
                        type=str, help='directory where the label files lie')
    parser.add_argument('in_model_dir', metavar='in_model_dir',
                        type=str, help='directory where the original models lie')
    parser.add_argument('--out_model_dir', metavar='out_model_dir',
                        type=str, default=None, help='directory where the models will be stored')
    args = parser.parse_args()

    initialize_gmm(args.n_dims, args.n_comps, args.feat_type, args.model_name, args.covar_type, args.model_dir)
