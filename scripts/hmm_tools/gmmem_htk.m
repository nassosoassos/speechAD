function [mix, options, errlog] = gmmem_htk(gmm, x, options)
% GMMEM_HTK EM algorithm for Gaussian Mixture Model using HTK
% 
% Prerequisites:
% matlab_htk_interface should be in path
%
% Notes: 
% The function does not really guarantee that the covariances will remain
% spherical if that is the case in the beginning.
% 

% Nassos Katsamanis, SAIL, 2010
% URL: http://sipi.usc.edu/~nkatsam

% Temporary directory structure needed for HTK
WORKING_DIR = 'tmp_htk'; 
DATA_DIR = fullfile(WORKING_DIR,'data','');
MODEL_DIR = fullfile(WORKING_DIR,'model','');
MODEL_NAME = 'gmm';
SCP_FILE = fullfile(WORKING_DIR, 'data.scp');
DATA_KIND = 'USER';
MAX_N_SAMPLES_FILE = 5000;

% Check that inputs are consistent
errstring = consist(gmm, 'gmm', x);
if ~isempty(errstring)
  error(errstring);
end

[ndata, xdim] = size(x);

% Sort out the options
if (options(14))
  niters = options(14);
else
  niters = 20;
end

display = options(1);
store = 0;
if (nargout > 2)
  store = 1;	% Store the error values to return them
end
test = 0;
if options(3) > 0.0
  test = 1;	% Test log likelihood for termination
end

check_covars = 0;
MIN_COVAR = 0;
if options(5) >= 1
  if display >= 0
    disp('check_covars is on');
  end
  check_covars = 1;	% Ensure that covariances don't collapse
  MIN_COVAR = eps;	% Minimum singular value of covariance matrix
  init_covars = mix.covars;
end


% Create the corresponding HTK model
orig_covar_type = gmm.covar_type;
gmm.vector_type = DATA_KIND;
gmm.name = MODEL_NAME;
[htk_model_def, htk_model_macros] = gmm_matlab2htk(gmm);

% Create the necessary directory structure
[s, m] = mkdir(WORKING_DIR);
[s, m] = mkdir(DATA_DIR);
[s, m] = mkdir(MODEL_DIR);

% Writing the GMM in HTK file format
htk_model_file = fullfile(MODEL_DIR, MODEL_NAME);
fmodel_handle = fopen(htk_model_file,'w');
fprintf(fmodel_handle,'%s%s',htk_model_macros, htk_model_def);
fclose(fmodel_handle);

% Writing the label file
htk_lab_file = fullfile(WORKING_DIR, [MODEL_NAME,'.lab']);
flab_handle = fopen(htk_lab_file,'w');
fprintf(flab_handle, '%s\n', MODEL_NAME);
fclose(flab_handle);

% Writing the data file
% Create multiple files when many samples are given
s_fid = fopen(SCP_FILE,'w');
n_samples = size(x,1);
i_sample = 1;
while i_sample<n_samples
    e_sample = min(i_sample + MAX_N_SAMPLES_FILE-1, n_samples);
    htk_data_file = fullfile(DATA_DIR, sprintf('gmm_%i.mfc',i_sample));
    writeHTK(htk_data_file, x(i_sample:e_sample,:), 0.01, htk_kind_code(DATA_KIND));
    fprintf(s_fid,'%s\n',htk_data_file);
    i_sample = e_sample+1;
end
fclose(s_fid);    

% Running EM using HTK Tool HRest
cmd = sprintf('HRest -A -T 1 -H %s -i %d -m 1 -v %f -S %s %s',...
    htk_model_file, niters, MIN_COVAR, SCP_FILE, MODEL_NAME);
[hrest_status, hrest_results] = system(cmd);
if ~isempty(regexpi(hrest_results,'Error|USAGE'))
    error(hrest_results);
end

logprob_info = regexp(hrest_results, 'LogProb\s+at\s+iter\s+\d+\s+=\s+(?<logprob>[-\d.]+)','names');
errlog = zeros(1, niters);
for iter_counter = 1:length(logprob_info)
    errlog(iter_counter) = -str2double(logprob_info(iter_counter).logprob);
end

% Reading back the GMM 
mix = gmm_htk2matlab(htk_model_file);
mix.type = 'gmm';

if strcmp(gmm.covar_type,'full')
    mix.covar_type = 'full';
    covar = mix.covars;
    for mix_comp_counter = 1:mix.ncentres
        mix.covars(:,:,mix_comp_counter) = inv(covar(:,:,mix_comp_counter));
    end

end

options(8) = -sum(log(gmmprob(mix, x)));

