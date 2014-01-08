function [Pmiss Pfa FramePercentage] = VadClassify(TrainFilesPaths,TestFilesPaths,Feature,ModelFilePath,Classifier,Parameter,MedianFilterLength,TaskId)
% runs K-NN for Voice Activity Detection
% TrainFilesPath    : 4-column input file of training file path
% TestFilesPath     : 4-column input file of testing file path
%                     For both files, the first column contains the audio
%                     file path, the second contains the annotation
%                     file path, the third contains the result file
%                     path and the fourth the feature file path.
% FeaturesStructName: Name under which the features have been save in a
%                     structure from Matlab
% ModelFilePath     : The path to save the constructed model.
% Classifier        : The classifier to use ('gmm','knn')
% Parameter         : Number of nearest neighbors used for K-NN (scalar value)
%                     or number of different mixtures for GMMs (vector value)
% MedianFilterLength: Length of median filtering to smooth 0/1 decision.
% TaskId            : ID of task for DARPA evaluation
% FramePercentage   : The percentage of correctly classified frames over
%                     all frames
%                       -for KNN, for all number of neighbors between
%                        [1,...,Parameter]. Results in a 1xParameter dimensional vector.
%                       -for GMMs, for all number of mixtures given in the
%                        vector. Results in a 1xlength(Parameter) dimensional vector.
% Pfa               : False alarm rate according to DARPA evaluation script
% Pmiss             : Missing rate according to DARPA evaluation script

% frame step for including frames in training
TrainFrameStep = 10;
TestFrameStep = 10;

% TODO: define them as optional arguments
% feature extraction window and step parameters
Shift = 0.01;
WindowLength = 0.02;
AnnotationShift = 0.1;

% path for DARPA evaluation script
EvalScriptPath = '/home/theodora/RATS/RES_v1-2/RES_1-2_ScoringEngine.jar';
% input file for DARPA evaluation script (3 columns: wav,annotation,result)
DARPAEvalInputFile = '/tmp/DARPA_input';
% directory of audio files (just something to use for DARPA evaluation script)
AudioDir = '/home/theodora/RATS/sad_ldc2011e86_v2/data/train';
% working directory of DARPA evaluation script
WorkingDir = '/tmp/';

% find annotation, result and feature files and store them into cell
% arrays
TrainFilesFid = fopen(TrainFilesPaths);
[TrainFileNames] = textscan(TrainFilesFid,'%s\t%s\t%s\t%s\n');
AnnotationPathsTrain = TrainFileNames{2};
FeaturePathsTrain = TrainFileNames{4};
fclose(TrainFilesFid);

TestFilesFid = fopen(TestFilesPaths);
[TestFileNames] = textscan(TestFilesFid,'%s\t%s\t%s\t%s\n');
WavPathsTest = TestFileNames{1};
AnnotationPathsTest = TestFileNames{2};
ResultPathsTest = TestFileNames{3};
FeaturePathsTest = TestFileNames{4};
fclose(TestFilesFid);

tic
% collect data for training
TrainData = cell(1,length(AnnotationPathsTrain));
TrainLabels = cell(1,length(AnnotationPathsTrain));
for j = 1:length(AnnotationPathsTrain)
    if strcmp(Feature,'MFCCs')
        AudioFeatures = load(FeaturePathsTrain{j});
        SequenceLength = size(AudioFeatures.Cepstra,2);
        TrainData{j} = AudioFeatures.Cepstra(1:6,1:TrainFrameStep:end);
    elseif strcmp(Feature(1:4),'LTSV')
        AudioFeatures = load(FeaturePathsTrain{j});
        AudioFeatures = AudioFeatures';
        SequenceLength = size(AudioFeatures,2);
        TrainData{j} = AudioFeatures(:,1:TrainFrameStep:end);
    end
    SpeechSegments = load(AnnotationPathsTrain{j});
    BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
    TrainLabels{j} = BinaryLabels(1:TrainFrameStep:end)+1; % increase labels by 1, because of standard input in Classify fuction
end

% collect data for testing
TestData = cell(1,length(AnnotationPathsTest));
TestLabels = cell(1,length(AnnotationPathsTest));
for j = 1:length(AnnotationPathsTest)
    if strcmp(Feature,'MFCCs')
        AudioFeatures = load(FeaturePathsTest{j});
        SequenceLength = size(AudioFeatures.Cepstra,2);
        TestData{j} = AudioFeatures.Cepstra(1:6,1:TestFrameStep:end);
    elseif strcmp(Feature(1:4),'LTSV')
        AudioFeatures = load(FeaturePathsTest{j});
        AudioFeatures = AudioFeatures';
        SequenceLength = size(AudioFeatures,2);
        TestData{j} = AudioFeatures(:,1:TestFrameStep:end);
    end
    SpeechSegments = load(AnnotationPathsTest{j});
    BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
    TestLabels{j} = BinaryLabels(1:TestFrameStep:end)+1; % increase labels by 1, because of standard input in Classify fuction
end

% check for NaN elements, if they exist, eliminate the file
TrainDataFixed = cell(1,0);
TrainLabelsFixed = cell(1,0);
AnnotationPathsTrainFixed = cell(1,0);
FeaturePathsTrainFixed = cell(1,0);
for j = 1:length(TrainData)
    if sum(sum(isnan(TrainData{j})))==0
        TrainDataFixed{end+1} = TrainData{j};
        TrainLabelsFixed{end+1}= TrainLabels{j};
        AnnotationPathsTrainFixed{end+1} = AnnotationPathsTrain{j};
        FeaturePathsTrainFixed{end+1} = FeaturePathsTrain{j};
    else
        fprintf('Warning: Training sample %d omitted because of NaN elements: %s\n',j,AnnotationPathsTrain{j});
    end
end

TestDataFixed = cell(1,0);
TestLabelsFixed = cell(1,0);
AnnotationPathsTestFixed = cell(1,0);
FeaturePathsTestFixed = cell(1,0);
WavPathsTestFixed = cell(1,0);
ResultPathsTestFixed = cell(1,0);
for j = 1:length(TestData)
    if sum(sum(isnan(TestData{j})))==0
        TestDataFixed{end+1} = TestData{j};
        TestLabelsFixed{end+1}= TestLabels{j};
        AnnotationPathsTestFixed{end+1} = AnnotationPathsTest{j};
        FeaturePathsTestFixed{end+1} = FeaturePathsTest{j};
        WavPathsTestFixed{end+1} = WavPathsTest{j};
        ResultPathsTestFixed{end+1} = ResultPathsTest{j};
    else
        fprintf('Warning: Testing sample %d omitted because of NaN elements: %s\n',j,AnnotationPathsTest{j});
    end
end

fprintf('Train and test data collected in %fsec.\n',toc);

% Classification
tic
[ClassProbabilities] = Classify(TrainDataFixed,TestDataFixed,TrainLabelsFixed,TestLabelsFixed,Classifier,Parameter,'/tmp/',ModelFilePath);
fprintf('Classifier run in %fsec.\n',toc);

% compute class decision
[ClassDecision] = ConvertClassProbabilitiesToClassDecision(ClassProbabilities, MedianFilterLength);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [ClassDecision2] = Classify(TrainData,TestData,TrainLabels,TestLabels,Classifier,Parameter,'/tmp/',ModelFilePath,1,0);
% [ClassDecision3] = Classify(TrainData,TestData,TrainLabels,TestLabels,Classifier,Parameter,'/tmp/',ModelFilePath,0,1);
% FramePercentage2 = sum(sum(cell2mat(ClassDecision2)'==cell2mat(TestLabels')))/length(cell2mat(TestLabels));
% FramePercentage3 = sum(sum(cell2mat(ClassDecision3)'==cell2mat(TestLabels')))/length(cell2mat(TestLabels));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% compute success percentage
NumberOfParameters = length(Parameter);
FramePercentage = sum(cell2mat(ClassDecision')==repmat(cell2mat(TestLabelsFixed)',1,NumberOfParameters))/length(cell2mat(TestLabelsFixed));

% convert class decision and labels into 0/1s (instead of 1/2s) and
% write results to predefined files for best neighbor
[maxval OptimalParameterPos] = max(FramePercentage);
format = '%d\n';
for i = 1:length(ClassDecision)
    ClassDecision{i} = ClassDecision{i}-1;
    TestLabelsFixed{i} = TestLabelsFixed{i}-1;
    fid = fopen(ResultPathsTestFixed{i},'w');
    fprintf(fid,format,ClassDecision{i}(:,OptimalParameterPos)');
    fclose(fid);
end
fprintf('Optimal Parameter = %d\n',Parameter(OptimalParameterPos));

% run DARPA evaluation script
fid = fopen(DARPAEvalInputFile,'w');
AnnotationPathsTestDARPA = cell(1,length(AnnotationPathsTestFixed));
for i = 1:length(AnnotationPathsTestDARPA)
    AnnotationPathsTestDARPA{i} = regexprep(AnnotationPathsTestFixed{i}, 'txt_S', 'txt');
    fprintf(fid,'%s\t%s\t%s\n',WavPathsTest{i},AnnotationPathsTestDARPA{i},ResultPathsTestFixed{i});
end
fclose(fid);
[Pmiss Pfa] = FindPercentageFromResultFiles(DARPAEvalInputFile,EvalScriptPath,AudioDir,WorkingDir,TaskId,AnnotationShift,WindowLength);