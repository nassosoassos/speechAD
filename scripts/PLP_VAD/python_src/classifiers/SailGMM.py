'''
Created on Jan 5, 2012

Generate GMM-distributed data and store them in HTK-formatted binary files.

@author: nassos
'''
import math
from xml.dom.minidom import parse, parseString
import os
import random
import numpy as np

from sklearn.mixture import GMM

class SailGMM(GMM):
    """
    Extending the scikits-learn GMM class to add support for reading from
    a file and customize the GMM initialization
    """
    def __init__(self, n_components=1, cvtype='diag', random_state=None,
                 thresh=1e-2, min_covar=1e-3, name='gmm', means=None,
                 covars=None, weights=None, definition_file=None,
                 definition_string = None, definition_dom = None):
        gmm_props = None
        if definition_file != None:
            if os.path.exists(definition_file):
                gmm_props = SailGMM.gmm_props_from_file(definition_file)
        elif definition_string != None:
            gmm_props = SailGMM.gmm_props_from_string(definition_string)
        elif definition_dom != None:
            gmm_props = SailGMM.gmm_props_from_xml(definition_dom)

        if gmm_props != None:
            n_components = len(gmm_props["mixes"])
            n_dims = gmm_props["n_dims"]
            means = np.zeros((n_components, n_dims))
            weights = gmm_props["priors"]
            name = gmm_props["name"]

            # Make the assumption that the covariance type is the same for all
            # the GMM components.
            cvtype = gmm_props["mixes"][0]["cov_type"]
            if cvtype == 'spherical':
                covars = np.zeros(n_components)
            elif cvtype == 'diag':
                covars = np.zeros((n_components, n_dims))
            elif cvtype == 'tied':
                covars = np.zeros((n_dims, n_dims))
            elif cvtype == 'full':
                covars = np.zeros((n_components, n_dims, n_dims))

            for mx_counter in range(n_components):
                c_mix = gmm_props["mixes"][mx_counter]
                means[mx_counter,:] = c_mix["mean"].transpose()
                if cvtype == 'spherical':
                    covars[mx_counter] = c_mix["covariance"]
                elif cvtype == 'diag':
                    covars[mx_counter,:] = c_mix["covariance"].transpose()
                elif cvtype == 'tied':
                    covars = c_mix["covariance"]
                elif cvtype == 'full':
                    covars[mx_counter,:,:] = c_mix["covariance"]

        super(SailGMM, self).__init__(n_components=n_components, cvtype=cvtype,
                                      random_state=random_state, thresh=thresh,
                                      min_covar=min_covar)
        self.name = name
        if means != None:
            self.means = means
        if covars != None:
            self.covars = covars
        if weights != None:
            self.weights = weights

    @staticmethod
    def generate_gmm_distributed_data(gmm_file, samples_file, n_samples):
        gmm = SailGMM.gmm_props_from_file(gmm_file)
        n_comps = gmm["mixes"].size
        n_samples_computed = 0
        samples = np.zeros(gmm["n_dims"],n_samples)
        for counter in range(n_comps):
            n_samples_cmpn = math.ceil(gmm["priors"][counter]*n_samples)
            if n_samples_computed+n_samples_cmpn>n_samples:
                n_samples_cmpn = n_samples - n_samples_computed
                start_ind = n_samples_computed
                end_ind = n_samples_computed+n_samples_cmpn-1
                samples[:,start_ind:end_ind] = np.random.multivariate_normal(gmm,gmm["n_dims"]*n_samples_cmpn)
                n_samples_computed += n_samples_cmpn

    @staticmethod
    def gmm_props_from_xml(dom_gmm, convert_covars=False):
        '''
        The xml dom looks like this:
        <gmm>
          <mixture>
            <weight> 0.3 </weight>
            <mean> -1 -1 -1 1 1 1 1 1 1 1</mean>
            <covariance type="diag">
             1 2 3 1 1 1 1 1 1 1
            </covariance>
          </mixture>
          <mixture>
            <weight> 0.3 </weight>
            <mean> -2 -3 12 1 1 1 1 1 1 1</mean>
            <covariance type="diag">
              1 2 3 1 1 1 1 1 1 1
            </covariance>
          </mixture>
          <mixture>
            <weight> 0.4 </weight>
            <mean> -9 4 -5 1 1 1 1 1 1 1</mean>
            <covariance type="diag">
              1 2 3 1 1 1 1 1 1 1
            </covariance>
          </mixture>
        </gmm>
        '''
        gmm = {}
        comps = []
        weights = []
        mixes = dom_gmm.getElementsByTagName("mixture")
        gmm["name"] = dom_gmm.getElementsByTagName("name")[0].firstChild.data.strip()
        for mx in mixes:
            mix = {}
            weights.append(float(mx.getElementsByTagName("weight")[0].firstChild.data))
            mix["mean"] = np.fromstring(mx.getElementsByTagName("mean")[0].firstChild.data,dtype=np.float64,sep=" ")
            cov_info = mx.getElementsByTagName("covariance")[0]
            cov_type = cov_info.getAttribute("type")
            cov_arr = np.fromstring(cov_info.firstChild.data,dtype=np.float64,sep=" ")
            n_dims = mix["mean"].size
            if convert_covars:
                if cov_type == "diag":
                    mix["covariance"] = np.diag(cov_arr)
                elif cov_type == "spherical":
                    mix["covariance"] = cov_arr[0]*np.eye(n_dims)
                else:
                    mix["covariance"] = np.reshape(cov_arr,n_dims,n_dims)
            else:
                mix["covariance"] = cov_arr

            mix["cov_type"] = cov_type
            comps.append(mix)

        gmm["n_dims"] = n_dims
        gmm["mixes"] = comps
        gmm["priors"] = np.divide(weights,sum(weights))
        return(gmm)

    @staticmethod
    def gmm_props_from_string(definition_string, convert_covars=False):
        dom_gmm = parseString(definition_string)
        return SailGMM.gmm_props_from_xml(dom_gmm, convert_covars=convert_covars)

    @staticmethod
    def gmm_props_from_file(gmm_file, convert_covars=False):
        '''
        Read a .xml formatted file defining a GMM.
        '''
        dom_gmm = parse(gmm_file)
        return(SailGMM.gmm_props_from_xml(dom_gmm, convert_covars=convert_covars))

    @staticmethod
    def gmm_list_from_string(definition_string):
        return(SailGMM.gmm_list_from_xml(parseString(definition_string)))

    @staticmethod
    def gmm_list_from_file(model_file):
        return(SailGMM.gmm_list_from_xml(parse(model_file)))

    @staticmethod
    def gmm_list_from_xml(dom_models):
        '''
        Read a .xml file defining multiple gmms
        '''
        gmm_list = []
        gmms = dom_models.getElementsByTagName("gmm")
        for gm in gmms:
            gmm_list.append(SailGMM(definition_dom=gm))

        return(gmm_list)

    @staticmethod
    def sample_gmms(gmm_list, n_samples):
        n_gmms = len(gmm_list)
        gmm_ids = range(n_gmms)
        min_n_seg_samples = 3
        max_n_seg_samples = n_samples / 3

        n_dims = gmm_list[0].means.shape[1]
        samples = np.zeros((n_samples, n_dims))
        assert(n_gmms<128)
        labels = np.zeros(n_samples, dtype=np.int8)
        generated_samples = 0
        seg_start = 0
        seg_end = 0
        while generated_samples < n_samples:
            seg_length = min(random.randrange(min_n_seg_samples,max_n_seg_samples),
                             n_samples-generated_samples)
            seg_end = seg_start + seg_length
            seg_id = random.sample(gmm_ids,1)[0]
            labels[seg_start:seg_end] = seg_id*np.ones(seg_length)
            samples[seg_start:seg_end,:] = gmm_list[seg_id].rvs(n_samples=seg_length)
            seg_start = seg_end
            generated_samples += seg_length

        return(samples, labels)


if __name__ == '__main__':
    a = SailGMM()
    print a
    b = SailGMM(definition_file='/Users/nassos/Documents/Work/robust_speech_processing/RATS/scripts/test_gmms/gmm_simple.xml')
    print b.means
    print b.covars
    print b.weights
    samples = b.rvs(10)
    print samples

