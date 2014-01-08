function vadsohn_wrapper(input_audio_file, output_file)
%VADSOHN_WRAPPER - Function to estimate VAD using Voicebox's vadsohn VAD
%
% Usage:
%   vadsohn_wrapper(input_audio_file, output_file)
%
% Description:
% Provide a standardized interface to the vadsohn function. The output file
% will be in .txt format with two columns, one with the frame numbers and
% the second with the corresponding vad decision, either 0 or 1.
%
% Input:
% input_audio_file : the audio filename (absolute path, wav file expected)
% output_file : the output filename (absolute path)
% 
FRAMERATE = 0.01; % In seconds


% Check whether the input file exists
if ~exist(input_audio_file,'file')
    error_msg = strcat('The given audio file %s does not exist.', input_audio_file);
    error(error_msg);
end

% Create the output file 
[out_fid, fopen_msg] = fopen(output_file,'w');
if out_fid == -1
    error(fopen_msg);
end

try
    [audio_signal, Fs] = wavread(input_audio_file);
    signal_duration = length(audio_signal)/Fs;
    expected_n_frames = floor(signal_duration/FRAMERATE);

    % Apply the VAD
    vad_options.ti = FRAMERATE;
    va_decision = vadsohn(audio_signal, Fs, 't', vad_options);
    n_frames = size(va_decision,1);
    
    if (abs(n_frames-expected_n_frames)>2)
        error('Unexpected number of frames at the VAD output');
    end
    if va_decision(1,3) ~= 1 && va_decision(1,3) ~= 0
        error('Unexpected vad output, not 0 or 1');
    end
    
    for i_frame = 1:n_frames
        fprintf(out_fid,'%i %i\n',i_frame,va_decision(i_frame,3));
    end
catch ME
    fclose(out_fid);
    error(ME.message);
end

fclose(out_fid);
