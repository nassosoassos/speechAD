function [TimeStamps] = ConvertFramesToTimeStamps(FrameSequence,Shift,WindowLength,MaxSequenceLengthSec)
% [TimeStamps] = ConvertFramesToTimeStamps(FrameSequence,Shift,WindowLength,MaxSequenceLengthSec)
%
% Converts a sequence of 0/1 into time stamps (in seconds)
% Input:    FrameSequnce is a 1-dimensional vector of 0/1s
%           Shift is the shift of each frame in seconds
%           WindowLength is the length of each window in seconds
%           MaxSequenceLengthSec is the actual length of the frame sequence
%           in seconds. This argument is optional.
% Output:   sequence of time stamps
% Example:  [TimeStamps] = ConvertFramesToTimeStamps([0 0 1 1 0 1 1],0.01,0.03)

TimeStamps=[];

if nargin<3,
    fprintf('Number of input arguments should be 4 or 3.\n');
    return;
end

if sum(FrameSequence==1)+sum(FrameSequence==0)~=length(FrameSequence)
    fprintf('Error: Input must have 0/1 indices for speech/non-speech.\n');
    return;
end

% get starting and ending speech indices
StartingSpeech=find(diff(FrameSequence)>0);
EndingSpeech=find(diff(FrameSequence)<0);

% in case everything is speech
if isempty(EndingSpeech) && isempty(StartingSpeech) && sum(FrameSequence)==length(FrameSequence)
    TimeStamps(1,1) = 0;
    TimeStamps(1,2) = (length(FrameSequence)-1)*Shift;
    return
end

% special case: check if there is speech in the first frame
index=1;
if FrameSequence(1)==1,
    TimeStamps(1,1)=0;
    TimeStamps(1,2)=(EndingSpeech(1)-1)*Shift + WindowLength/2;
    index=index+1;
end

% general case: convert start and end frames to time-stamps
TimeStamps(index:index+length(StartingSpeech)-1,1) = (StartingSpeech(1:end))*Shift + WindowLength/2;
TimeStamps(index:length(EndingSpeech),2) = (EndingSpeech(index:end)-1)*Shift + WindowLength/2;


% special case: check if there is speech in the end frame
if FrameSequence(end)==1,
    if nargin <4
        TimeStamps(end,2)=(length(FrameSequence)-1)*Shift;
    elseif nargin == 4,
        TimeStamps(end,2)=MaxSequenceLengthSec;
    end
end
