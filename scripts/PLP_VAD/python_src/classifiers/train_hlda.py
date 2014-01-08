'''
Created on Dec 28, 2011

@author: nassos
'''
from htk.herest import herest
import os
import re

# Number of iterations needed for HLDA
max_iterations = 5

# String to define the global class
# needed for HLDA by HTK
global_class_string='''~b \"global\"
<MMFIDMASK> *
<PARAMETERS> MIXBASE
<NUMCLASSES> 1
<CLASS> 1 {{*.state[{0}].mix{1}'''



def get_global_class_string(n_mixes, n_states=1):
    '''
    Could not find another way to modify the number of
    mixture components required in the string defining the
    global class.
    '''
    if n_states==1:
        state_str = '2';
    else:
        state_str = '{{2-{0}}}'.format(str(n_states+1))
    g_string = global_class_string.format(state_str,'[1-{0}]}}'.format(str(n_mixes)))
    return g_string

def config_hlda(n_nuisance_dims,max_iterations):
    conf = {}
    conf['HADAPT:TRANSKIND'] = 'SEMIT'
    conf['HADAPT:USEBIAS'] = 'FALSE'
    conf['HADAPT:BASECLASS'] = 'global'
    conf['HADAPT:SPLITTHRESH'] = 0.0
    conf['HADAPT:MAXXFORMITER'] = 100
    conf['HADAPT:MAXSEMITIEDITER'] = max_iterations
    conf['HADAPT:SEMITIED2INPUTXFORM'] = 'TRUE'
    conf['HADAPT:NUMNUISANCEDIM'] = n_nuisance_dims
    conf['HADAPT:SEMITIEDMACRO'] = 'HLDA'
    conf['HADAPT:TRACE'] = 61
    conf['HMODEL:TRACE'] = 512

    return conf

def train_hlda(model_list, n_nuisance_dims, training_list, lab_dir, model_dir, min_var_macro=None, n_states=1):
    '''
    Estimate HLDA transformation by following the guidelines in the HTK
    tutorial (HTKBook, p.49 'Semi-Tied and HLDA transforms'). Found that
    HLDA estimation in HTK v3.4-1 only works
    '''
    # Find the number of GMM components
    model_file = os.path.join(model_dir,'newMacros')
    assert(os.path.exists(model_file))
    m_file = open(model_file,'r')
    n_comps = 0
    for ln in m_file:
        m = re.match('<NUMMIXES>\s+(\d+)',ln)
        if m:
            n_comps = m.group(1)
            break
    if n_comps==0:
        raise Exception

    # Write global class file
    class_filename = os.path.join(model_dir,'global')
    class_file = open(class_filename,'w')
    class_file.write(get_global_class_string(n_comps, n_states))
    class_file.close()

    # Write configuration file
    config_filename = os.path.join(model_dir,'config.hlda')
    hlda_conf_map = config_hlda(n_nuisance_dims, max_iterations)
    write_htk_config(hlda_conf_map, config_filename)

    # Call HERest to estimate the HLDA transform
    args = ['-H',os.path.join(model_dir,'newMacros'),'-u','s',
            '-C',config_filename,'-L',lab_dir,'-K',model_dir,
            '-J',model_dir,'-S',training_list,'-M',model_dir,model_list]
    if min_var_macro != None and os.path.exists(min_var_macro):
        args.insert(0, '-H')
        args.insert(1, min_var_macro)
    herest(args)

    # Reiterate after the HLDA estimation
    min_var_macro = os.path.join(model_dir,'vFloors')
    if min_var_macro != None and os.path.exists(min_var_macro):
        args = ['-H', os.path.join(model_dir,'newMacros'),'-C', config_filename,
                '-L', lab_dir, '-S', training_list, '-H', min_var_macro,
                '-M', model_dir, model_list]
    else:
        args = ['-H', os.path.join(model_dir,'newMacros'),'-C', config_filename,
                '-L', lab_dir, '-S', training_list, '-M', model_dir, model_list]

    for _ in range(1, 3):
        herest(args)

def write_htk_config(conf_map, conf_filename):
    conf_file = open(conf_filename,'w')
    for att in conf_map.keys():
        conf_file.write('{0} = {1}\n'.format(att,str(conf_map[att])))

if __name__ == '__main__':
    pass
