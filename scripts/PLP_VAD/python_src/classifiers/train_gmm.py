'''
Created on Dec 28, 2011

@author: nassos
'''
import logging
import os
import string
import subprocess

from htk import hrest, herest
from my_utils import which
from classifiers import initialize_gmm

os.environ['PATH'] += os.pathsep + '/usr/local/bin'
hhed_bin = which.which('HHEd')

def hhed(args):
    cmd = args
    cmd.insert(0, hhed_bin)
    logging.debug(string.join(cmd,' '))
    logging.info(subprocess.check_call(cmd, stderr=subprocess.STDOUT))

def increase_n_components(n_mixes, model_name, model_file, training_list, lab_dir,
                          target_model_dir, min_var_macro_file=None, n_states=1, do_kmeans=False):
    '''
    Increase the number of GMM components
    '''
    hhed_cfg_file = os.path.join(target_model_dir,'hhed.cfg')
    hhed_cfg = open(hhed_cfg_file,'w')
    hhed_cfg.write('KEEPDISTINCT=TRUE')
    hhed_cfg.close()

    hhed_cmd_file = os.path.join(target_model_dir,'hhed.cmd')
    hhed_cmd = open(hhed_cmd_file,'w')
    if n_states > 1:
        hhed_cmd.write('MU {0} {{{1}.state[2-{2}].mix}}\n'.format(str(n_mixes), model_name, str(n_states+1)))
    else:
        hhed_cmd.write('MU {0} {{{1}.state[2].mix}}\n'.format(str(n_mixes), model_name))
    hhed_cmd.close()

    model_list_file = os.path.join(target_model_dir,'model.list')
    model_list = open(model_list_file,'w')
    model_list.write(model_name)
    model_list.close()

    model_dir = os.path.split(model_file)[0]
    hhed_args = ['-C', hhed_cfg_file, '-d', model_dir, '-M', target_model_dir, hhed_cmd_file, model_list_file]
    hhed(hhed_args)

    if do_kmeans == True:
       initialize_gmm.initialize_gmm_kmeans(model_name, training_list, lab_dir, target_model_dir, target_model_dir)
    train_gmm(model_file=os.path.join(target_model_dir, model_name),
              training_list=training_list,
              lab_dir=lab_dir,
              out_model_dir=target_model_dir,
              n_train_iterations=30,
              min_var_macro=min_var_macro_file)

def prepare_vfloor_macros_file(model_file, vfloors_file):
    # Merge variance floors into model file
    model_f = open(model_file, 'r')
    model_info = model_f.readlines()
    model_f.close()
    min_var_macro = open(vfloors_file,'r')
    min_var_info = min_var_macro.readlines()
    min_var_macro.close()
    min_var_macro = open(vfloors_file, 'w')
    n_lines = len(model_info)
    var_written = False
    ln = 0
    while ln < n_lines and model_info[ln].find("~h") == -1:
        min_var_macro.write(model_info[ln])
        ln += 1
    min_var_macro.writelines(min_var_info)

def train_gmm(model_file, training_list, lab_dir, out_model_dir, n_train_iterations=20, min_var_macro=None):
    '''
    Train single GMM
    '''
    model_dir, model_name = os.path.split(model_file)
    model_name = os.path.splitext(model_name)[0]
    if min_var_macro != None and os.path.exists(min_var_macro):
        args = ['-u','mvw','-S',training_list,'-L',lab_dir, '-l', model_name,
                '-H', min_var_macro, '-M', model_dir, '-m',str(1), model_file]
    else:
        args = ['-u','mvw','-S',training_list,'-L',lab_dir,'-l', model_name,
                '-M',model_dir, '-m', str(1), model_file]

    hrest.hrest_serial(args)

def train_gmm_set(model_list, training_list, lab_dir, model_dir, orig_model_file, n_train_iterations=10,
              min_var_macro=None, update_transitions=False):
    '''
    Train GMM files for different classes.

    Input:
    model_list : file with names of the models to be trained (one per line)
                 the model names should match the labels in the annotations
    training_list : list of training feature files (absolute paths)
    orig_dir : directory where the initial model files lie (HTK mmf-formatted file is required)
    model_dir : directory where the output models will be stored
    n_train_iterations : number of Baum-Welch iterations
    min_var_macro : file with the minimum variance macros (vFloors)
    '''
    if min_var_macro != None and os.path.exists(min_var_macro):
        args = ['-u','mvwt','-S',training_list,'-L',lab_dir,'-H',orig_model_file,
                '-H', min_var_macro, '-M', model_dir, '-m',str(1),model_list]
    else:
        args = ['-u','mvwt','-S',training_list,'-L',lab_dir,'-H',orig_model_file,
                '-M',model_dir, '-m', str(1), model_list]

    herest.herest(args)
    model_file = os.path.join(model_dir,'newMacros')
    if not os.path.exists(model_file):
        model_file = os.path.join(model_dir,'hmmdef')
    assert(os.path.exists(model_file))

    #print "Trained for a single iteration"

    if min_var_macro != None and os.path.exists(min_var_macro):
        args = ['-u','mvwt','-S',training_list,'-L',lab_dir,'-H',model_file,
                '-H',min_var_macro, '-M',model_dir, '-m',str(1),model_list]
    else:
        args = ['-u','mvwt','-S',training_list,'-L',lab_dir,'-H',model_file,
                '-M',model_dir, '-m',str(1),model_list]

    # Ideally a convergence criterion should be applied instead
    for iter in range(1,n_train_iterations):
        #print "Iteration {0}/{1}".format(iter, n_train_iterations)
        #print args
        herest.herest(args)

def train_gmm_file(model_file, model_list, training_list, lab_dir, model_dir, n_iterations=3):
    args = ['-S',training_list,'-L',lab_dir,'-H',model_file,'-m','-M',model_dir,str(1),model_list]
    for _ in range(1,n_iterations):
        herest(args)


if __name__ == '__main__':
    pass
