'''
Created on Dec 29, 2011

@author: nassos
'''
from classifiers import evaluate_gmm_classifier

def vad_gmm_evaluate_frames(ref_annotations_list, hyp_annotations_list, samp_period, model_list, mode='sequence'):
    m_list = open(model_list,'r')
    labels = []
    for m in m_list:
        labels.append(m.rstrip('\r\n'))
    conf_matrix = evaluate_gmm_classifier.evaluate_results_list(ref_annotations_list, hyp_annotations_list,
                                                  samp_period, labels, mode)
    return(conf_matrix)

if __name__ == '__main__':
    pass
