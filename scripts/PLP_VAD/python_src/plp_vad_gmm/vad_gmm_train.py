'''
Created on Dec 29, 2011

@author: nassos
'''
import os
import logging
import shutil

from plp_vad_gmm import vad_extract_features as vef
from classifiers import train_gmm_classifier

def vad_gmm_train(audio_list, annotations_dir, model_list, feature_type,
                          n_coeffs_per_frame, acc_frames, working_dir, apply_hlda=False,
                          hlda_nuisance_dims=0, n_gmm_comps=None, acc_frame_shift=1,
                          n_train_iterations=10, samp_period=0.01, segment_boundaries=[1/3.0, 2/3.0]):
    logging.info('Number of gmm components: '+str(n_gmm_comps))

    features_dir = os.path.join(working_dir,feature_type)
    acc_features_dir = features_dir+'_frames'
    fea_file_list = os.path.join(working_dir,'feature_files.list')
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # Feature extraction
    vef.fea_extract(audio_list,feature_type,n_coeffs_per_frame,features_dir, samp_period)
    vef.create_corresponding_list_assert(audio_list,features_dir,fea_file_list,'fea')
    if not os.path.exists(acc_features_dir):
        os.makedirs(acc_features_dir)
    vef.accumulate_feature_vectors_parallel(fea_file_list, acc_frames,
                                            acc_features_dir, acc_frame_shift)
    vef.create_corresponding_list_assert(audio_list,acc_features_dir,fea_file_list,'fea')
    #vef.split_feature_files_labels(fea_file_list, annotations_dir, label_list,
    #                               boundaries=segment_boundaries)

    # GMM training
    if n_gmm_comps==None:
        raise ValueError('Automatic determination of the number of the GMM components not implemented')

    model_dir = os.path.join(working_dir,'models')
    train_gmm_classifier.train_M_sized_gmm_classifier_incrementally(n_gmm_comps, n_coeffs_per_frame,
                                                                    feature_type, fea_file_list, model_list,
                                                                    annotations_dir, model_dir, apply_hlda,
                                                                    hlda_nuisance_dims, n_train_iterations)
    model_files = []
    m_f = os.path.join(model_dir,'newMacros')
    model_files.append(m_f)
    m_f_no_hlda = m_f+"_no_hlda"
    model_files.append(m_f_no_hlda)

    assert(os.path.exists(m_f))
    return model_files

if __name__ == '__main__':
    pass
