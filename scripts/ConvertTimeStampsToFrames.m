function [FrameSequence] = ConvertTimeStampsToFrames(TimeStamps,Shift,WindowLength,SequenceLength)
% [FrameSequence] = ConvertTimeStampsToFrames(TimeStamps,Shift,WindowLength,SequenceLength)
% Converts time stamps (in seconds) into a sequence of 0/1
% Input:    FrameSequnce is a sequence of time stamps (2-dimensional vector of start and end times)
%           Shift is the shift of each frame in seconds
%           WindowLength is the length of each window in seconds
%           SequenceLengthSec is the actual length of the frame sequence
%           in frames (in order to complete with 0s at the end).
% Output:   1-dimensional vector of 0/1s
% Example:  [FrameSequence] = ConvertFramesToTimeStamps([0.1 2.2; 3.5 4; 5.6 9.7],0.01,0.03)

TimeStampsInFrames = min(SequenceLength,max(1,round((TimeStamps-(WindowLength-Shift))/Shift)));

FrameSequence = zeros(1,SequenceLength);

for i = 1:size(TimeStampsInFrames,1)
    FrameSequence(TimeStampsInFrames(i,1):TimeStampsInFrames(i,2)) = 1;
end