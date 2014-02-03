'''
Created on Dec 29, 2011

@author: nassos
'''
import os
from plp_vad_gmm import vad_extract_features as vef
from classifiers.test_gmm_classifier import test_gmm_classifier

def vad_gmm_list(audio_list, model_list, model_file, feature_type, 
            n_coeffs_per_frame, acc_frames, results_dir, working_dir, samp_period, win_length):
    features_dir = os.path.join(working_dir,feature_type)
    acc_features_dir = features_dir+'_frames'
    fea_file_list = os.path.join(working_dir,'feature_files.list')
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(acc_features_dir):
        os.makedirs(acc_features_dir)
    vef.fea_extract(audio_list,feature_type,n_coeffs_per_frame,features_dir,samp_period, win_length)
    vef.create_corresponding_list_assert(audio_list,features_dir,fea_file_list,'fea')
    vef.accumulate_feature_vectors(fea_file_list,acc_frames,acc_features_dir)
    vef.create_corresponding_list_assert(audio_list,acc_features_dir,fea_file_list,'fea')
      
    test_gmm_classifier(fea_file_list, model_list, model_file, results_dir)    
   
if __name__ == '__main__':
    pass