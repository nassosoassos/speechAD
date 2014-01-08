function [mix, options, errlog] = gmmem_init_htk_inc(gmm_model, x, options)
% GMMEM_INIT_HTK Gaussian Mixture Model training using HTK. Initialization
% is achieved incrementally.
% 
% Prerequisites:
% matlab_htk_interface should be in path
%
% Notes: 
% The function does not really guarantee that the covariances will remain
% spherical if that is the case in the beginning.
% 

% Nassos Katsamanis, SAIL, 2011
% URL: http://sipi.usc.edu/~nkatsam

% Temporary directory structure needed for HTK
WORKING_DIR = fullfile(pwd,'tmp_htk'); 
DATA_DIR = fullfile(WORKING_DIR,'data','');
MODEL_DIR = fullfile(WORKING_DIR,'model','');
MODEL_NAME = 'gmm';
MODEL_LIST = fullfile(WORKING_DIR,'models.list');
SCP_FILE = fullfile(WORKING_DIR, 'data.scp');
DATA_KIND = 'USER';
MAX_N_SAMPLES_FILE = 5000;

% Check that inputs are consistent
errstring = consist(gmm_model, 'gmm', x);
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
MIN_COVAR = 0.001;
if options(5) >= 1
  if display >= 0
    disp('check_covars is on');
  end
  check_covars = 1;	% Ensure that covariances don't collapse
  MIN_COVAR = eps;	% Minimum singular value of covariance matrix
  init_covars = mix.covars;
end

% Prototype GMM
proto_ncentres = 1;
proto_input_dim = gmm_model.nin;
proto = gmm(proto_input_dim, proto_ncentres, gmm_model.covar_type);
proto.vector_type = DATA_KIND;
proto.name = MODEL_NAME;
[htk_model_def, htk_model_macros] = gmm_matlab2htk(proto);

% Create the necessary directory structure
[s, m] = mkdir(WORKING_DIR);
[s, m] = mkdir(DATA_DIR);
[s, m] = mkdir(MODEL_DIR);

% Write model list
fid = fopen(MODEL_LIST,'w');
if fid<0 
    error('Unable to open %s for writing',MODEL_LIST);
end
fprintf(fid,'%s\n',MODEL_NAME);
fclose(fid);

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
    lab_file = fullfile(DATA_DIR, sprintf('gmm_%i.lab',i_sample));
    lab_fid = fopen(lab_file,'w');
    fprintf(lab_fid,'%s\n',MODEL_NAME);
    fclose(lab_fid);
    i_sample = e_sample+1;
end
fclose(s_fid);    

cmd = sprintf('HCompV -v %f -m -S %s %s',...
    MIN_COVAR,SCP_FILE, htk_model_file);
[status, hcompv_results] = system(cmd);

if ~isempty(regexpi(hcompv_results,'Error|USAGE'))
    error(hcompv_results);
end

% Increasing the number of Gaussians
source_dir = MODEL_DIR;
for nm = 2:gmm_model.ncentres
    target_dir = fullfile(MODEL_DIR, num2str(nm));
    [s,w] = mkdir(target_dir);
    
    edCmdfile = fullfile(target_dir,'split.hed');
    edCmd_fid = fopen(edCmdfile, 'w');
    fprintf(edCmd_fid, 'MU %d {*.state[2].mix}\n',nm);
    fclose(edCmd_fid);
    hhed_config = fullfile(target_dir,'hhed_conf');
    hhed_config_fid = fopen(hhed_config,'w');
    fprintf(hhed_config_fid,'KEEPDISTINCT=TRUE\n');
    fclose(hhed_config_fid);    
    cmd = sprintf('HHEd -A -C %s -d %s -M %s %s %s', hhed_config, source_dir, target_dir,...
         edCmdfile, MODEL_LIST);
    [hhed_status, hhed_results] = system(cmd);
    if ~isempty(regexpi(hhed_results,'Error|USAGE'))
          error(hhed_results);
    end
    htk_model_file = fullfile(target_dir,MODEL_NAME);
    cmd = sprintf('HRest -A -T 1 -H %s -i %d -m 1 -v %f -S %s %s',...
        htk_model_file, niters, MIN_COVAR, SCP_FILE, MODEL_NAME);
    disp(cmd);
    [hrest_status, hrest_results] = system(cmd);
    if ~isempty(regexpi(hrest_results,'Error|USAGE'))
        error(hrest_results);
    end

    logprob_info = regexp(hrest_results, 'LogProb\s+at\s+iter\s+\d+\s+=\s+(?<logprob>[-\d.]+)','names');
    errlog = zeros(1, niters);
    for iter_counter = 1:length(logprob_info)
        errlog(iter_counter) = -str2double(logprob_info(iter_counter).logprob);
    end
    source_dir = target_dir;
end


% Reading back the GMM 
mix = gmm_htk2matlab(htk_model_file);
mix.type = 'gmm';

if strcmp(gmm_model.covar_type,'full')
    mix.covar_type = 'full';
    covar = mix.covars;
    for mix_comp_counter = 1:mix.ncentres
        mix.covars(:,:,mix_comp_counter) = inv(covar(:,:,mix_comp_counter));
    end
end

options(8) = -sum(log(gmmprob(mix, x)));

