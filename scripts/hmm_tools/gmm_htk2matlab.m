function gmm = gmm_htk2matlab(htk_model_file)
% GMM_HTK2MATLAB Read an HTK-formatted GMM file
%
% Usage:
%       gmm = gmm_htk2matlab(htk_model_file)
%
% Description:
% The htk_model_file is an HTK model parameter file.

% Nassos Katsamanis, SAIL, 2010
% URL: http://sipi.usc.edu/~nkatsam

fmodel_file = fopen(htk_model_file,'r');

vector_info_pattern = '<VecSize>\s*(?<vecSize>\d+)\s*<NULLD>\s*<(?<vecType>[\w\_]+)>';
name_info_pattern = '~h\s+"(?<name>[\w\d]+)"';
num_mixes_info_pattern = '<NumMixes>\s*(?<numMixes>[\d]+)';

vector_info = []; name_info = []; num_mixes_info = [];

% Vector information retrieval, i.e., type and dimension
while isempty(vector_info)
    cline = fgetl(fmodel_file);
    if ~ischar(cline)
        break;
    end
    vector_info = regexpi(cline, vector_info_pattern, 'names');
end    
if ~isempty(vector_info)
    gmm.nin = str2double(vector_info.vecSize);
    gmm.vector_type = vector_info.vecType;
end

% Name information retrieval, i.e., name of the model
while isempty(name_info)
    cline = fgetl(fmodel_file);
    if ~ischar(cline)
        error('No models found.');
    end
    name_info = regexpi(cline, name_info_pattern, 'names');
end    
if ~isempty(name_info)
    gmm.name = name_info.name;
end

% Number of mixture components retrieval
while isempty(num_mixes_info) 
    cline = fgetl(fmodel_file);
    if ~ischar(cline)
        error('Badly formatted HTK model parameter file');
    end
    num_mixes_info = regexpi(cline, num_mixes_info_pattern, 'names');    
end
if ~isempty(num_mixes_info)
    gmm.ncentres = str2double(num_mixes_info.numMixes);
end


% Mixture retrieval
weight_info_pattern = '<Mixture>\s*[\d]+\s+(?<weight>[e-\+\d\.]+)';
mean_info_pattern = '(?<mean>[e-\+\d\.]+)';
var_info_pattern = '(?<variance>[e-\+\d\.]+)';
n_mix_comps = gmm.ncentres;
for mix_comp_counter = 1:n_mix_comps    
    % Mixture weight retrieval
    weight_info = []; mean_info = []; var_info = [];
    while isempty(weight_info)
        cline = fgetl(fmodel_file);
        if ~ischar(cline)
            error('Badly formatted HTK model parameter file');
        end
        weight_info = regexpi(cline, weight_info_pattern, 'names');
    end
    if ~isempty(weight_info) 
        gmm.priors(mix_comp_counter) = str2double(weight_info.weight);
    end
    
    % Mean retrieval
    cline = fgetl(fmodel_file);
    while  (ischar(cline) && ~regexpi(cline,'Mean'))
        cline = fgetl(fmodel_file);
    end    
    if ~ischar(cline)
        error('Badly formatted HTK model parameter file');
    end      
    while isempty(mean_info)
        cline = fgetl(fmodel_file);
        if ~ischar(cline)
            error('Badly formatted HTK model parameter file');
        end
        mean_info = regexpi(cline, mean_info_pattern, 'names');        
    end
    if ~isempty(mean_info)
        for dim_counter = 1:gmm.nin
            gmm.centres(mix_comp_counter, dim_counter) = str2double(mean_info(dim_counter).mean); 
        end
    end
    
    % Variance retrieval
    % Mean retrieval
    cline = fgetl(fmodel_file);
    while  (ischar(cline) && ~regexpi(cline,'Variance|InvCovar'))
        cline = fgetl(fmodel_file);
    end    
    if ~ischar(cline)
        error('Badly formatted HTK model parameter file');
    end      
    % Consider same covariance matrix structure for all mixture components
    if regexpi(cline,'Variance')
        % Diagonal Covariane Matrix has been found
        gmm.covar_type = 'diag';
        while isempty(var_info)
            cline = fgetl(fmodel_file);
            if ~ischar(cline)
                error('Badly formatted HTK model parameter file');
            end
            var_info = regexpi(cline, var_info_pattern, 'names');
        end
        if ~isempty(var_info)
            for dim_counter = 1:gmm.nin
                gmm.covars(mix_comp_counter, dim_counter) = str2double(var_info(dim_counter).variance);
            end
        end
    else
        % Full covariance matrix has been found
        gmm.covar_type = 'invcovar';
        utri = zeros(gmm.nin, gmm.nin);

        for dim_counter = 1:gmm.nin
            cline = fgetl(fmodel_file);
            if ~ischar(cline) 
                error('Badly formatted HTK model parameter file');
            end
            var_info = regexpi(cline, var_info_pattern, 'names');
            if isempty(var_info)
                error('Badly formatted HTK model parameter file');
            end                
            for col_counter = 1:(gmm.nin-dim_counter+1)
                utri(dim_counter, dim_counter+col_counter-1) = str2double(var_info(col_counter).variance);
            end            
        end
        gmm.covars(:,:,mix_comp_counter) = (utri+utri.')/2;
    end
    gmm.type = 'gmm';
end
fclose(fmodel_file);
