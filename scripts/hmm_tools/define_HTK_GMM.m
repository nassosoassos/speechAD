function define_HTK_GMM(n_dims, n_comps, cov_type, feature_type, name, directory)
%DEFINE_HTK_GMM - Define a GMM and write it in HTK format
% 
% Usage:
%   define_HTK_GMM(n_dims, n_mixtures, cov_type, name, file)
% 
% Description:
% Given the number of mixtures and the number of dimensions, write a
% prototype GMM file. 
% 
% 
mix = gmm(n_dims, n_comps, cov_type);
mix.name = name;
mix.vector_type = feature_type;

[htk_model_def, htk_model_macros] = gmm_matlab2htk(mix);
htk_model_file = fullfile(directory,name);

fmodel_handle = fopen(htk_model_file,'w');
fprintf(fmodel_handle,'%s%s',htk_model_macros, htk_model_def);
fclose(fmodel_handle);
