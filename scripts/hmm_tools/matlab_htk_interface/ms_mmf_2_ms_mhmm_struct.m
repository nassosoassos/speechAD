function ms_mhmm_struct = ms_mmf_2_ms_mhmm_struct(fname, use_log, scale_alpha)
% MS_MMF_2_MS_MHMM_STRUCT - Reads a multistream hmm from an HTK file
% 
%   ms_mhmm_struct = ms_mmf_2_ms_mhmm_struct(fname)
%   
% Description:
% 
% Example: 
% 
% Project: audiovisual speech recognition
% See also: 
%   

% Step 1: Read a (multi-stream) HMM model from an htk model file
mmfs_struct=msMmf2masv(fname);

S = length(mmfs_struct);

adjust_scale = exist('scale_alpha');
if ~exist('use_log') 
    %use_log=true; 
    use_log = false;
end

% dbstop masv2kpm;

% Step 2: Put each stream's hmm independently into K. Murphy's format
for i=1:S
  [prior, transmat, mu, Sigma, mixmat, sweights, name, cov_type] = ...
      masv2kpm(mmfs_struct(i));
  if (use_log), [prior,transmat,mixmat]=logTransMat(prior,transmat,mixmat);end
  if adjust_scale
    inv_scale_alpha_i = 1 ./ scale_alpha{i};
    mu = diag(inv_scale_alpha_i)*mu;
    Sigma = diag(inv_scale_alpha_i.^2)*Sigma;
  end  
  temp = kpm2struct(prior, transmat, mu, Sigma, mixmat, sweights, name);
  temp.use_log = use_log;
  kpms_struct(i) = temp;
end

% Step 3: 
ms_mhmm_struct = kpms_struct2ms_mhmm_struct(kpms_struct);
