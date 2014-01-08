function [header, data] = readHTK(inputFileName)
% READHTKHEADER - Read the header of a file in HTK format
% 
%   [header, data] = readHTK(inputFileName)
%   
% Description:
% Follow the guidelines in HTKBook and read the header of an
% HTK-formatted file. 
% Input Arguments:
% 
% Output Arguments:
% 
% Example: 
% 
% Project: HTK tools
% See also: 
%   
 
% Copyright: Nassos Katsamanis, CVSP Group, NTUA
% URL: http://cvsp.cs.ntua.gr/~nassos
% Created: 04/07/2005
fid = fopen(inputFileName, 'r', 'l');
if fid==-1 
  error(['Cannot open ', inputFileName]);
end

nSamples = fread(fid, 1, 'int32');
sampPeriod = fread(fid, 1, 'int32');
sampSize = fread(fid, 1, 'int16');

if nSamples<0 || sampPeriod<0 || sampSize<0 || sampSize>2000
  fclose(fid);
  fid = fopen(inputFileName, 'r','b');
  nSamples = fread(fid, 1, 'int32');
  sampPeriod = fread(fid, 1, 'int32');
  sampSize = fread(fid, 1, 'int16');
end
parmKind = fread(fid, 1, 'int8');
qualifier = fread(fid, 1, 'int8');

header.nSamples = nSamples;
header.sPeriod = sampPeriod;
header.sampSize = sampSize;
header.parmKind = parmKind;
header.qualifier = qualifier;

tmp = fread(fid, Inf, 'float32');

data = reshape(tmp, sampSize/4, nSamples);

fclose(fid);
