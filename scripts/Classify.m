function [ClassProbabilities] = Classify(TrainFeaturesCellPerFile,TestFeaturesCellPerFile,TrainLabelsCellPerFile,TestLabelsCellPerFile,Classifier,Parameter,WorkingDir,ModelFilePath)
% classification using different classifiers (GMMs, K-NN)
% TrainFeaturesCellPerFile    : cell of training features, each having a matrix of size (dxN)
% TestFeaturesCellPerFile     : cell of testing features, each having a matrix of size (dxN)
% TrainLabelsCellPerFile   : cell of of training labels, each having a vector of
%                 size(1xN) or (Nx1), with values 1,2,3,...
% TestLabelsCellPerFile    : cell of of testing labels, each having a vector of
%                 size(1xN) or (Nx1), with values 1,2,3,...
% Classifier    : what kind of classifier to use('gmm','knn','knn-mahalanobis')
% Parameters    : vector of number of gaussians for GMM
%               : or number of neighbors for K-NN
% WorkingDir    : our working directory
% ModelFilePath : for K-NN, the path to save the constructed model file
%               : for GMMs, a cell of paths to save the constructed model files
% ClassProbabilities    : probabilities of classes for test data, cell of 
%                         matrices each of dimension (CxNxParameter)
% In the above, d is the dimensionality, N is the number of samples, TotalNumberOfClasses is
% the number of classes

% TODO: Seperate path folder for matlab scripts

% TODO: check if nargin is correct
if nargin~=8
    fprintf('Error: Wrong number of arguments.\n');
    return
end

% check if ann_sample directory exists, if not exit
AnnSampleFileAbsolutePath = '/usr/bin/ann_sample';
if strcmp(Classifier,'knn') && ~exist(AnnSampleFileAbsolutePath,'file')
    fprintf('Error: Function ann_sample does not exist.\n');
    return
end

% check if grep directory exists, if not exit
GrepFileAbsolutePath = '/bin/grep';
if strcmp(Classifier,'knn') && ~exist(GrepFileAbsolutePath,'file')
    fprintf('Error: Function grep does not exist.\n'); 
    return
end

% check if proper classifier is set as input, if not exit
if ~(strcmp(Classifier,'gmm') || strcmp(Classifier,'knn') || ...
     strcmp(Classifier,'knn-mahalanobis'))
    fprintf('Error: Classifier provided is not supported.\n');
    return
end

% check if working directory exists, make it otherwise
if ~exist(WorkingDir,'dir')
    mkdir(WorkingDir);
end

% merge train and test feature and labels into single matrices
TrainingFeaturesMatrix = cell2mat(TrainFeaturesCellPerFile);
TrainingLabelsMatrix = cell2mat(TrainLabelsCellPerFile);
TestingFeaturesMatrix = cell2mat(TestFeaturesCellPerFile);
TestingLabelsMatrix = cell2mat(TestLabelsCellPerFile);

% check size of labels, if otherwise, convert them to be a row vector
if size(TrainingLabelsMatrix,1)>1; TrainingLabelsMatrix = TrainingLabelsMatrix'; end
if size(TestingLabelsMatrix,1)>1; TestingLabelsMatrix = TestingLabelsMatrix'; end


% TODO: check number of samples in data and labels
if size(TrainingFeaturesMatrix,2)~=size(TrainingLabelsMatrix,2)
    if size(TrainingFeaturesMatrix,1)==size(TrainingLabelsMatrix,2)
        TrainingFeaturesMatrix = TrainingFeaturesMatrix';
        fprintf('Warning: Training data has been transposed to have equal samples with labels\n');
    else
        fprintf('Error: Unequal number of samples in training data and labels.\n');
        return
    end
end
if size(TestingFeaturesMatrix,2)~=size(TestingLabelsMatrix,2)
    if size(TestingFeaturesMatrix,1)==size(TestingLabelsMatrix,2)
        TestingFeaturesMatrix = TestingFeaturesMatrix';
        fprintf('Warning: Testing data has been transposed to have equal samples with labels\n');
    else
        fprintf('Error: Unequal number of samples in testing data and labels.\n');
        return
    end
end

% check dimensionality of train and test data
if size(TrainingFeaturesMatrix,1)~=size(TestingFeaturesMatrix,1)
    fprintf('Error: Unequal number of features for training and testing data.\n');
    return
end

% find number of classes
TotalNumberOfClasses = length(unique(TrainingLabelsMatrix));

% TODO: Check all features in range 1:TotalNumberOfClasses
for i = 1:TotalNumberOfClasses
    if sum(TrainingLabelsMatrix==i)==0
        fprintf('Error: Wrong data labels. There are classes omitted in the training data.\n');
        return
    end
end

% count and save the number of samples for each cell on the test set
% and form accumulative counter
AccumSamplesPerFileVector = [1 zeros(1,length(TestFeaturesCellPerFile))];
for i = 2:length(TestFeaturesCellPerFile)+1
    AccumSamplesPerFileVector(i) = AccumSamplesPerFileVector(i-1)+size(TestFeaturesCellPerFile{i-1},2);
end

% number of test frames and test files
TestFilesNumber = length(TestFeaturesCellPerFile);
TestFramesNumber = size(TestingFeaturesMatrix,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if MatlabKNN
%     ClassProbabilitiesMatrix = knnclassify(TestingFeaturesMatrix',TrainingFeaturesMatrix',TrainingLabelsMatrix');
%     % split total class probability matrix for testing data into cells of matrices
%     ClassProbabilities = cell(1,size(TestFeaturesCellPerFile));
%     for i = 1:TestFilesNumber
%         ClassProbabilities{i} = ClassProbabilitiesMatrix(AccumSamplesPerFileVector(i):AccumSamplesPerFileVector(i+1)-1);
%     end
%     return
% end
% 
% if AndreasKNN
%     [ClassProbabilitiesMatrix final_accuracy]  =knn_class(TrainingFeaturesMatrix',TestingFeaturesMatrix',TrainingLabelsMatrix,TestingLabelsMatrix,1);
%     ClassProbabilities = cell(1,size(TestFeaturesCellPerFile));
%     for i = 1:TestFilesNumber
%         ClassProbabilities{i} = ClassProbabilitiesMatrix(AccumSamplesPerFileVector(i):AccumSamplesPerFileVector(i+1)-1);
%     end
%     return
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if strcmp(Classifier(1:3),'knn')  % K-NN classifier
    
    % define classifier parameters
    NearestNeighbors = Parameter;
    
    % data dimensionality
    DataDim = size(TrainingFeaturesMatrix,1);
    
    fprintf('data dimensionality = %d\n',DataDim);  % TODO: erase that
    
    % whiten data
    if strcmp(Classifier,'knn-mahalanobis')
        TrainDataCovariance = cov(TrainingFeaturesMatrix');
        UpperTrainCovarianceFactTerm = chol(inv(TrainDataCovariance));
        TrainingFeaturesMatrix = UpperTrainCovarianceFactTerm*TrainingFeaturesMatrix;
        TestingFeaturesMatrix = UpperTrainCovarianceFactTerm*TestingFeaturesMatrix;
    end
    
    % write test and train data into a temporary file
    format = repmat('%f ',1,size(TrainingFeaturesMatrix,1)); 
    format(end:end+1) = '\n';
    fid = fopen(fullfile(WorkingDir,'train'),'w');
    fprintf(fid,format,TrainingFeaturesMatrix);
    fclose(fid);
    fid = fopen(fullfile(WorkingDir,'test'),'w');
    fprintf(fid,format,TestingFeaturesMatrix);
    fclose(fid);

    % call K-NN function from command line toolbox and clear additional text with grep
    cmd = sprintf('%s -d %d -nn %d -max %d -df %s -qf %s | %s -v "(" | %s -v "NN" | %s -v Data > %s',AnnSampleFileAbsolutePath,DataDim,max(NearestNeighbors),size(TrainingFeaturesMatrix,2)+TestFramesNumber,fullfile(WorkingDir,'train'),fullfile(WorkingDir,'test'),GrepFileAbsolutePath,GrepFileAbsolutePath,GrepFileAbsolutePath,ModelFilePath);
    system(cmd);
    disp('ann_sample finished.');
    
    % clear feature matrices and cells to save memory
    clear TrainFeaturesCellPerFile TrainingFeaturesMatrix TestFeaturesCellPerFile TestingFeaturesMatrix;
    
    % split model file into separate models, in order to laod more quickly
    cmd = sprintf('split -l %d  %s %s_split_',30000*max(NearestNeighbors),ModelFilePath,ModelFilePath);
    system(cmd);
    
    % load results from separate model files and and find nearest neighbors
% % % % % % % %     Results = load(ModelFilePath);
% % % % % % % %     NearestNeighbor = buffer(Results(:,2),NearestNeighbors);
    
    [ModelFileDir ModelFileName ModelFileExt] = fileparts(ModelFilePath);
    ResultsFiles = dir(fullfile(ModelFileDir,sprintf('%s_split_*',ModelFileName)));
    ResultsFilesIds = [];
    for i = 1:length(ResultsFiles)
        ResultsFilesIds = [ResultsFilesIds; strtok(fliplr(ResultsFiles(i).name),'_')];
    end
    ResultsFilesIds = fliplr(ResultsFilesIds);
    ResultsFilesIdsNumerical = 26*(ResultsFilesIds(:,1)-double('a'))+(ResultsFilesIds(:,2)-double('a'));    % TODO: make it to work for more than 2 characters per alphabetical value
    ResultsFilesIdsNumericalSorted = sort(ResultsFilesIdsNumerical);
    ResultsFilesIdsSorted = [char(floor((ResultsFilesIdsNumericalSorted/26))+char('a')) char(mod(ResultsFilesIdsNumericalSorted,26)+double('a'))];

    ClassProbabilitiesMatrix = zeros(TotalNumberOfClasses,TestFramesNumber,length(NearestNeighbors));
    for i = 1:size(ResultsFilesIdsSorted,1)
        SplitModelFileName = fullfile(ModelFileDir,sprintf('%s_split_%s',ModelFileName,ResultsFilesIdsSorted(i,:)));
        ResultsPart = load(SplitModelFileName);
        NearestNeighbor = buffer(ResultsPart(:,2),max(NearestNeighbors));
        for c = 1:TotalNumberOfClasses
            for n = 1:length(NearestNeighbors)
                if i==size(ResultsFilesIdsSorted,1)
                    ClassProbabilitiesMatrix(c,30000*(i-1)+1:end,n) = sum(TrainingLabelsMatrix(NearestNeighbor(1:NearestNeighbors(n),:)+1)==c,1)/NearestNeighbors(n);
                else
                    ClassProbabilitiesMatrix(c,30000*(i-1)+1:30000*i,n) = sum(TrainingLabelsMatrix(NearestNeighbor(1:NearestNeighbors(n),:)+1)==c,1)/NearestNeighbors(n);
                end
            end
        end
        system(sprintf('rm %s',SplitModelFileName));
    end
    
    % clear memory
    clear ResultsPart NearestNeighbor;
    
    disp('Class Probabilities calculated.');
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Results = [];
%     for i = 1:size(ResultsFilesIdsSorted,1)
%         SplitModelFileName = fullfile(ModelFileDir,sprintf('%s_split_%s',ModelFileName,ResultsFilesIdsSorted(i,:)));
%         ResultsPart = load(SplitModelFileName);
%         Results = [Results; ResultsPart(:,2)];
%         system(sprintf('rm %s',SplitModelFileName));
%     end
%     NearestNeighbor = buffer(Results,NearestNeighbors);
%     disp('Nearest Neigbors loaded.');
%     
%     % clear memory
%     clear Results ResultsPart;
%     
%     % find probabilities for each class
%     ClassProbabilitiesMatrix = zeros(TotalNumberOfClasses,TestFramesNumber,NearestNeighbors);
%     for c = 1:TotalNumberOfClasses
%         for n = 1:NearestNeighbors
%             ClassProbabilitiesMatrix(c,:,n) = sum(TrainingLabelsMatrix(NearestNeighbor(1:n,:)+1)==c,1)/n;
%         end
%     end
%     disp('Class Probabilities calculated.');
%     
%     % clear memory
%     clear NearestNeighbor;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

elseif strcmp(Classifier,'gmm')   % GMM classifier
    
    % define number of possible gaussian mixtures
    NumberOfMixtures = Parameter;
    
    % find probabilities for each class
    ClassProbabilitiesMatrix = zeros(TotalNumberOfClasses,size(TestingFeaturesMatrix,2),length(NumberOfMixtures));
    
    % for all input gaussian mixtures
    for k = 1:length(NumberOfMixtures)
        
        % create cell GMM models for all classes
        GmmModels = cell(1,TotalNumberOfClasses);
        
        % create, initialize and train GMMs for each class
        for c = 1:TotalNumberOfClasses
            % take data corresponding to the specific class
            ClassTrainData = TrainingFeaturesMatrix(:,TrainingLabelsMatrix==c);
            % TODO: Parameter Struct 'full', 'diagonal', etc
            % TODO: WokingDir option, test HTK files HREST etc path and
            % TODO: give waring for both parallel and sequential
            GmmModels{c} = gmm(size(ClassTrainData,1), NumberOfMixtures(k), 'full');
            % initialize GMM
            KmeansInitializationOptions = foptions;
            KmeansInitializationOptions(14) = 50;	% number of iterations for k-means
            GmmModels{c} = gmminit(GmmModels{c}, ClassTrainData', KmeansInitializationOptions);
            % run EM
            EmOptions = zeros(1, 18);
            EmOptions(1)  = 1;		% Prints out error values.
            EmOptions(14) = 50;		% Number of iterations.
            [GmmModels{c}, EmOptions, ErrorLog] = gmmem_htk(GmmModels{c}, ClassTrainData', EmOptions);
        end
        
        % save GMM model into working directory
        save(ModelFilePath{k},'GmmModels');
        
        load(ModelFilePath{k},'GmmModels');
        % find probabilities for each class for a specific gaussian
        for c = 1:TotalNumberOfClasses
            ClassProbabilitiesMatrix(c,:,k) = (gmmprob(GmmModels{c}, TestingFeaturesMatrix'))';
        end
        
        fprintf('Mixture %d done\n',NumberOfMixtures(k));
        
    end
end


% split total class probability matrix for testing data into cells of matrices
ClassProbabilities = cell(1,TestFilesNumber);
for i = 1:TestFilesNumber
    ClassProbabilities{i} = ClassProbabilitiesMatrix(:,AccumSamplesPerFileVector(i):AccumSamplesPerFileVector(i+1)-1,:);
end
