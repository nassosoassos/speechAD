function [htk_model_def, htk_model_macros] = gmm_matlab2htk(gmm)
% GMM_MATLAB2HTK Converting the gmm for use with HTK
% 
% Usage: 
%      htk_model_def = gmm_matlab2htk(gmm)
% 
% Desctiption:
% The input gmm is in netlab format. 

% Nassos Katsamanis, SAIL, 2010
% URL: http://sipi.usc.edu/~nkatsam

covar_string = '';
switch gmm.covar_type
    case {'spherical','diag'}
        covar_string = 'DIAGC';
    case 'full'
        covar_string = 'FULLC';
end

htk_model_macros = sprintf('~o <VecSize> %d <NULLD><%s><%s>\n', gmm.nin, gmm.vector_type, covar_string);
htk_model_def = sprintf('~h "%s"\n', gmm.name);
htk_model_def = sprintf('%s<BeginHMM>\n', htk_model_def);
htk_model_def = sprintf('%s\t<NumStates> %d\n', htk_model_def, 3);

n_mix_comps = length(gmm.priors);
htk_model_def = sprintf('%s\t<State> 2 <NumMixes> %d\n', htk_model_def, n_mix_comps);

for mix_comp_counter = 1:n_mix_comps 
    htk_model_def = sprintf('%s\t\t<Mixture> %d %f\n', htk_model_def, mix_comp_counter, gmm.priors(mix_comp_counter));

    % Means
    htk_model_def = sprintf('%s\t\t\t<Mean> %d\n', htk_model_def, gmm.nin);
    mean_vec_str = mat2str(gmm.centres(mix_comp_counter,:));   
    mean_vec_str = regexprep(mean_vec_str,'[\[\]]','');
    htk_model_def = sprintf('%s\t\t\t\t%s\n', htk_model_def, mean_vec_str);
    
    % Variances
    switch gmm.covar_type
        case {'spherical','diag'}
            if strcmp(gmm.covar_type,'spherical')
                variance = gmm.covars(1, mix_comp_counter)*ones(1,gmm.nin);
            else
                variance = gmm.covars(mix_comp_counter,:);
            end
            var_str = mat2str(variance);
            var_str = regexprep(var_str,'[\[\]]','');
            htk_model_def = sprintf('%s\t\t\t<Variance> %d\n', htk_model_def, gmm.nin);
            htk_model_def = sprintf('%s\t\t\t\t%s\n', htk_model_def, var_str);
        case {'full','invcovar'}
            % We should store the upper triangular part of the inverse covariance matrix            
            if strcmp(gmm.covar_type,'full')
                mix_comp_covar = gmm.covars(:,:,mix_comp_counter);
                % Matrix inversion (May be slow, consider the 'invcovar' option)
                inv_mix_comp_covar = inv(mix_comp_covar);
            else
                inv_mix_comp_covar = gmm.covars(:,:,mix_comp_counter);
            end
            
            htk_model_def = sprintf('%s\t\t\t<InvCovar> %d\n', htk_model_def, gmm.nin);
            for dim_counter = 1:gmm.nin
                inv_covar_row = inv_mix_comp_covar(dim_counter,dim_counter:gmm.nin);
                inv_covar_row_str = mat2str(inv_covar_row);
                inv_covar_row_str = regexprep(inv_covar_row_str,'[\[\]]','');
                htk_model_def = sprintf('%s\t\t\t\t%s\n', htk_model_def, inv_covar_row_str);            
            end                                                            
        case 'ppca'
            disp('Covariance type conversion to HTK not implemented');
    end               
end

% Transition matrix (The GMM is defined as a degenerate HMM)
htk_model_def = sprintf('%s\t<TransP> %d\n', htk_model_def, 3);
trans_mat = zeros(3,3);
trans_mat(1,2) = 1; trans_mat(2,2) = 0.9; trans_mat(2,3) = 0.1;
trans_mat_str = mat2str(trans_mat);
trans_mat_str = regexprep(trans_mat_str,'[\[\]]','');
trans_mat_str = regexprep(trans_mat_str,';','\n');
htk_model_def = sprintf('%s%s\n', htk_model_def, trans_mat_str);

htk_model_def = sprintf('%s<EndHMM>\n', htk_model_def);


