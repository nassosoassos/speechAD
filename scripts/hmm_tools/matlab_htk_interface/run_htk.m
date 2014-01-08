function [status, result] = run_htk(tool, arguments, varargin)
% Run the specified HTK tool with the given arguments
if nargin>2
    tool = fullfile(varargin{1},tool);
end

command = sprintf('%s %s', tool, arguments);
[status, result] = system(command);