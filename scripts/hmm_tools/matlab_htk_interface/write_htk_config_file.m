function write_htk_config_file(params,config_file)
% Write the given parameters properly formated in the HTK configuration
% file
f_config_file = fopen(config_file,'w');

param_names = fieldnames(params);
n_params = length(param_names);

for k=1:n_params
    k_value = params.(param_names{k});
    if isnumeric(k_value)
        k_value = num2str(k_value);
    end
    if ~isempty(strfind(k_value,'..')) || ~isempty(strfind(k_value,'/'))
      continue;
    end
    fprintf(f_config_file,'%s\t=\t%s\n',param_names{k},k_value);
end
