function initialize_HTK_GMM(model_file, data_list, label_dir, out_model_dir)
% INITIALIZE_HTK_GMM - Initialize a GMM in HTK format using K-means
%
% Usage:
%   initialize_HTK_GMM(model_file, data_list, label_dir, out_model_dir)
%
% Description:
% Use the k-means implementation of MATLAB to initialize a GMM which is in
% HTK format.

% Nassos Katsamanis, SAIL, 2011
% URL: http://sipi.usc.edu/~nkatsam
MAX_N_MATRIX_ELMS = 700e6;
mix = gmm_htk2matlab(model_file);

% Collect all the available data for this model
gmm_name = mix.name;

d_list_fid = fopen(data_list,'r');
data_files_info = textscan(d_list_fid,'%s');
data_files = data_files_info{1};
n_data_files = length(data_files);
fclose(d_list_fid);

% Find out the sampling period
hd = readHTK(data_files{1});
samp_period = hd.sPeriod;
n_dims = hd.sampSize/4;

n_frames = 0;
info = struct('start_frames', cell(1,n_data_files),...
    'end_frames',cell(1,n_data_files));
n_segments = zeros(1,n_data_files);
for k=1:n_data_files
    [pth, b_name] = fileparts(data_files{k});
    lab_file = fullfile(label_dir, [b_name,'.lab']);
    l_fid = fopen(lab_file,'r');
    label_info = textscan(l_fid,'%d%d%s');
    fclose(l_fid);
    label_names = label_info{3};
    label_start = label_info{1};
    label_end = label_info{2};
    accepted_segments = find(strcmp(gmm_name, label_names));
    n_segments(k) = length(accepted_segments);
    
    if n_segments(k)>0
        start_frames = zeros(n_segments(k),1);
        end_frames = zeros(n_segments(k),1);
    else
        start_frames = [];
        end_frames = [];
    end
    for s = 1:n_segments(k)
        % Take into consideration that the first frame starts at time 0
        start_frame = round(label_start(accepted_segments(s))/samp_period) +1 ;
        end_frame = round(label_end(accepted_segments(s))/samp_period);
        n_frames = n_frames + end_frame - start_frame + 1;
        start_frames(s) = start_frame;
        end_frames(s) = end_frame;
    end
    info(k).start_frames = start_frames;
    info(k).end_frames = end_frames;            
end

n_elms = n_frames * n_dims;
if n_elms > MAX_N_MATRIX_ELMS
    resampling_factor = 3*ceil(n_elms/MAX_N_MATRIX_ELMS);
    n_frames = ceil(n_frames/double(resampling_factor)) + sum(n_segments);
    disp(sprintf('The data will be subsampled at a rate %f to avoid exceeding maximum memory capacity', 1/double(resampling_factor)));
else
    resampling_factor = 1;
end
tot_n_frames = n_frames;
gmm_data = zeros(tot_n_frames, n_dims);
cur_index = 1;
for k=1:n_data_files
    [hd, data] = readHTK(data_files{k});
    
    n_segments = length(info(k).start_frames);
    for s=1:n_segments
        s_frame = info(k).start_frames(s);
        e_frame = info(k).end_frames(s);
        frame_range = s_frame:resampling_factor:e_frame;
        seg_data = data(:, frame_range);
        n_frames = length(frame_range);
        gmm_data(cur_index:cur_index + n_frames-1,:) = seg_data.';
        cur_index = cur_index + n_frames;
    end
end
if cur_index ~= tot_n_frames
    disp(sprintf('Predicted number of frames: %d Actual number of frames: %d',tot_n_frames, cur_index));
end

options = foptions;
mix = gmminit(mix, gmm_data(1:cur_index-1,:), options);

[pth, m_file] = fileparts(model_file);
new_model_file = fullfile(out_model_dir, m_file);
[htk_model_def, htk_model_macros] = gmm_matlab2htk(mix);
fmodel_handle = fopen(new_model_file,'w');
fprintf(fmodel_handle,'%s%s',htk_model_macros, htk_model_def);
fclose(fmodel_handle);
