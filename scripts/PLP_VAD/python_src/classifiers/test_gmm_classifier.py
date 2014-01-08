'''
Created on Dec 28, 2011

@author: nassos
'''
import re
import os

from onyx.htkfiles import htkmmf, htkaudio

from htk.hvite import hvite
from htk.create_trivial_wordnet import create_trivial_wordnet
from htk.create_trivial_dict import create_trivial_dict
from htk import htkmfc

def score_feature_sequences_gmm(in_file_list, htk_model_file, out_dir):
    '''
    Read an HTK model file defining a set of GMMs and score the input feature files
    based on these GMMs
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Read ASCII mmf file
    htk_mmf = open(htk_model_file,'r')
    models, hmm_mgr, gmm_mgr  = htkmmf.read_htk_mmf_file(htk_mmf, log_domain=True)
    htk_mmf.close()

    # Read feature files and score
    in_list = open(in_file_list, 'r')
    for fl in in_list:
        fl = fl.rstrip('\r\n')
        print fl
        feature_file = open(fl,'r')
        features = htkaudio.read_htk_audio_file(feature_file)
        feature_file.close()
        scores = []
        n_models = gmm_mgr.num_models
        for md in range(n_models):
           model_scores = gmm_mgr.get_model(md).score(features)
           scores.append(model_scores)

        score_file = os.path.join(out_dir, os.path.splitext(os.path.split(feature_file)[1])[0]+'.sco')
        n_frames = len(scores[0])
        sco_fl = open(score_file,'w')
        for count in range(n_frames):
            for m_count in range(n_models):
                sco_fl.write('{} '.format(scores[m_count][count]))
            sco_fl.write('\n')
        sco_fl.close()

    in_list.close()

def create_corresponding_list_assert(in_file_list,out_dir,out_file_list,out_sfx):
    '''
    Change the extension of a filename, check if the new file exists
    and if yes write it in the out_file_list
    '''
    in_list = open(in_file_list,'r')
    out_list = open(out_file_list,'w')

    n_orig_files = 0
    n_final_files = 0
    for fl in in_list:
        n_orig_files += 1
        fname = os.path.splitext(os.path.split(fl.rstrip('\r\n'))[1])[0]
        out_file = fname+'.'+out_sfx
        if os.path.exists(os.path.join(out_dir,out_file)):
            out_list.write('%s.%s\n' % (fname, out_sfx))
            n_final_files += 1

    assert(n_orig_files==n_final_files)
    in_list.close()
    out_list.close()

def read_model_names_from_mmf(model_file):
    m_file = open(model_file,'r')
    model_labels = []
    for line in m_file:
        m = re.match(r"~h \"([^\"]+)\"",line)
        if m:
            model_labels.append(m.group(1))

    m_file.close()
    return model_labels

def test_gmm_classifier(test_list, model_list, model_file, results_dir, mode='sequence'):
    '''
    Test GMM classifier using HTK's tool HVite.
    '''
    m_list = open(model_list,'r')
    class_labels = []
    for l in m_list:
        l = l.rstrip('\r\n')
        class_labels.append(l)
    m_list.close()
    wdnet_file = os.path.join(results_dir, 'trivial_wdnet')
    create_trivial_wordnet(class_labels,wdnet_file, mode)
    dict_file = os.path.join(results_dir, 'trivial_dict')
    create_trivial_dict(class_labels,dict_file)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    hvite_args = ['-H',model_file,'-S',test_list,'-l',results_dir,
                  '-w',wdnet_file,dict_file,model_list]
    hvite(hvite_args)

    # Create list of results
    results_list = test_list+'.results'
    create_corresponding_list_assert(test_list,results_dir,results_list,'rec')

if __name__ == '__main__':
    pass
