function [ClassDecision] = ConvertClassProbabilitiesToClassDecision(ClassProbabilities, MedianFilterLength)
% Converts class probabilities into class decisions according to maximum
% probability rule and then performs median filtering
% ClassProbabilities    : probabilities of classes for test data, cell of 
%                         matrices each of dimension (CxNxParameter)
% MedianFilterLength    : size of median filtering
% ClassDecision         : decision according to maximum probability, cell
%                         of matrices each of dimension (NxParameter)
% C is the number of classes, N is the number of samples, Parameter is the
% number of parameters of the classifier

ClassDecision = cell(1,length(ClassProbabilities));
for i = 1:length(ClassProbabilities)
    [maxval indices] = max(ClassProbabilities{i},[],1);
    ClassDecision{i} = round(medfilt1(squeeze(indices),MedianFilterLength,[],1));
end