function []=VadClassifyRbFeat(DimReducTechnique,DimReducParam,NormBoolean,ResultsFileTaskId)

% feature extraction window and step parameters
AnnotationsShift = 0.1;
RbFeaturesShift = 0.1;

% path for DARPA evaluation script
EvalScriptPath = '/home/theodora/RATS/RES_v1-2/RES_1-2_ScoringEngine.jar';
% input file for DARPA evaluation script (3 columns: wav,annotation,result)
DARPAEvalInputFile = '/tmp/DARPA_input';
% directory of audio files (just something to use for DARPA evaluation script)
AudioDir = '/home/theodora/RATS/sad_ldc2011e86_v2/data/train';
% working directory of DARPA evaluation script
WorkingDir = '/tmp/';

% input directory of features
FeaturesDir = '/media/DATTRANSFER/RATS/Features/rbfeatmoving';

% input file of paths
InputFilesDir='/home/theodora/RATS/scripts/FilenamePaths/';

% model files directory
ModelFilesDir = '/media/DATTRANSFER/RATS/models/VadModels2';

% spec folder
SpecGramDir='/media/DATTRANSFER/RATS/Features/Spectrogram_128/';

% channels
Channels = {'A' 'B' 'C' 'D' 'F' 'G' 'H' 'src'};

% frame step for including frames in training
TrainFrameStep = 1;
TestFrameStep = 1;

% Classifier and classification parameters
Classifier = 'knn';
Parameter = [1 5 11 31];

% result directories
ResultsDir = strcat('/media/DATTRANSFER/RATS/results/rbfeatmoving_',Classifier);

% length of median filtering to smooth final labels
MedianFilterLength = 5;

XBandAlpha=[ 0  0.15  0.3 0.4  0.5 0.6 0.7 0.8 0.95];

YBandAlpha=[ 0  0.15  0.3 0.4  0.5 0.6 0.7 0.8 0.95	];

nBandsX=6;
nBandsY=4;

Moving=[1]; % Moving=[1 10 20];

AnalysisWindow=[20 10 30 50 75 ];

for ChIndx = 1:length(Channels)
    
    fprintf('\n**************************************************\n\n');
    fprintf('Channel %s\n',Channels{ChIndx});
    
    TrainFilesFid = fopen(fullfile(InputFilesDir,sprintf('%s_SmallScaleExperimentsTrain.txt',Channels{ChIndx})));
    TrainFiles = textscan(TrainFilesFid,'%s\t%s\t%s\t%s\n');
    TrainFilesAudio = TrainFiles{1};
    TrainFilesAnnotation = TrainFiles{2};
    fclose(TrainFilesFid);
    
    TrainData = cell(1,length(TrainFilesAnnotation));
    TrainLabels = cell(1,length(TrainFilesAnnotation));
    
    TestFilesFid = fopen(fullfile(InputFilesDir,sprintf('%s_SmallScaleExperimentsTest.txt',Channels{ChIndx})));
    TestFiles = textscan(TestFilesFid,'%s\t%s\t%s\t%s\n');
    TestFilesAudio = TestFiles{1};
    TestFilesAnnotation = TestFiles{2};
    fclose(TestFilesFid);
    
    TestData = cell(1,length(TestFilesAnnotation));
    TestLabels = cell(1,length(TestFilesAnnotation));
    TestFilesAnnotationDARPA = cell(1,length(TestFilesAnnotation));
    
    OverallResultsFid = fopen(fullfile(ResultsDir,'overall_results',sprintf('Ch%s_%s',Channels{ChIndx},ResultsFileTaskId)),'w');
    
    for MovingIndex = 1:length(Moving)
        for AnalysisWindowIndex = 1:length(AnalysisWindow)
            for XBandAlphaIndex = 1:length(XBandAlpha)
                for YBandAlphaIndex = 1:length(YBandAlpha)
                    
                    fprintf('Smooth = %d\t Analysis Win = %d\t AlphaX = %d\t AlphaY = %d\n',Moving(MovingIndex),AnalysisWindow(AnalysisWindowIndex),100*XBandAlpha(XBandAlphaIndex),100*YBandAlpha(YBandAlphaIndex));
                    
                    % analysis window length in seconds
                    WindowLength = AnalysisWindow(AnalysisWindowIndex)/100;
                    
                    % create results paths
                    ResultsFolder = fullfile(ResultsDir,ResultsFileTaskId,int2str(Moving(MovingIndex)),int2str(AnalysisWindow(AnalysisWindowIndex)),int2str(100*XBandAlpha(XBandAlphaIndex)),int2str(100*YBandAlpha(YBandAlphaIndex)));
                    if ~exist(ResultsFolder,'dir')
                        mkdir(ResultsFolder);
                    end
                    
                    tic
                    % train features and annotations
                    for FileIndex = 1:length(TrainFilesAudio)
                        [TrainFileDir TrainFileName] = fileparts(TrainFilesAudio{FileIndex});
                        % concatenate 1-dimensional Rb features
                        RbFeat = [];
                        for nBandsXIndex = 2:nBandsX-1
                            for nBandsYIndex = 1:nBandsY-2
                                FeatureParametersStr = strcat('rb',int2str(XBandAlpha(XBandAlphaIndex)*100) ,'_',int2str(nBandsXIndex),'_',int2str(nBandsX),'_',int2str(YBandAlpha(YBandAlphaIndex)*100),'_',int2str(nBandsYIndex),'_',int2str(nBandsY));
                                load(fullfile(FeaturesDir,int2str(Moving(MovingIndex)),int2str(AnalysisWindow(AnalysisWindowIndex)),...
                                    FeatureParametersStr,strcat(TrainFileName,'.mat')),'RbFeatTemp');
                                RbFeat = [RbFeat; squeeze(RbFeatTemp)'];
                            end
                        end
                        % save features and labels to cell
                        if NormBoolean
                            TrainData{FileIndex} = RbFeat(:,1:TrainFrameStep:end)./repmat(sum(RbFeat(:,1:TrainFrameStep:end,1)),size(RbFeat,1),1);
                        else
                            TrainData{FileIndex} = RbFeat(:,1:TrainFrameStep:end);
                        end
                        SpeechSegments = load(TrainFilesAnnotation{FileIndex});
                        SequenceLength = size(RbFeat,2);
                        BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,AnnotationsShift,WindowLength,SequenceLength);
                        TrainLabels{FileIndex} = BinaryLabels(1:TrainFrameStep:end)+1;
                        
%                         spec = load(fullfile(SpecGramDir,TrainFileName));
%                         subplot(3,1,1); imagesc(log(spec.Spectrogram(5:50,1:1000)));
%                         subplot(3,1,2); imagesc(TrainData{FileIndex}(:,1:100));
%                         subplot(3,1,3); plot(TrainLabels{FileIndex}(1:1000));
%                         pause
                    end
                    % test features and annotations
                    ResultPathsTest = cell(1,length(TestFilesAnnotation));
                    for FileIndex = 1:length(TestFilesAudio)
                        [TestFileDir TestFileName] = fileparts(TestFilesAudio{FileIndex});
                        % concatenate 1-dimensional Rb features
                        RbFeat = [];
                        for nBandsXIndex = 2:nBandsX-1
                            for nBandsYIndex = 1:nBandsY-2
                                FeatureParametersStr = strcat('rb',int2str(XBandAlpha(XBandAlphaIndex)*100) ,'_',int2str(nBandsXIndex),'_',int2str(nBandsX),'_',int2str(YBandAlpha(YBandAlphaIndex)*100),'_',int2str(nBandsYIndex),'_',int2str(nBandsY));
                                load(fullfile(FeaturesDir,int2str(Moving(MovingIndex)),int2str(AnalysisWindow(AnalysisWindowIndex)),...
                                    strcat('/rb',int2str(XBandAlpha(XBandAlphaIndex)*100) ,'_',int2str(nBandsXIndex),'_',int2str(nBandsX),'_',int2str(YBandAlpha(YBandAlphaIndex)*100),'_',int2str(nBandsYIndex),'_',int2str(nBandsY)),...
                                    strcat(TestFileName,'.mat')),'RbFeatTemp');
                                RbFeat = [RbFeat; squeeze(RbFeatTemp)'];
                            end
                        end
% %                         spec = load(fullfile(SpecGramDir,TestFileName));
% %                         subplot(2,1,1); imagesc(log(spec.Spectrogram(5:50,1:1000)));
% %                         subplot(2,1,2); imagesc(RbFeat(:,1:100));
% %                         pause
                        % save features and labels to cell
                        if NormBoolean
                            TestData{FileIndex} = RbFeat(:,1:TestFrameStep:end)./repmat(sum(RbFeat(:,1:TestFrameStep:end,1)),size(RbFeat,1),1);
                        else
                            TestData{FileIndex} = RbFeat(:,1:TestFrameStep:end);
                        end
                        SpeechSegments = load(TestFilesAnnotation{FileIndex});
                        SequenceLength = size(RbFeat,2);
                        BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,AnnotationsShift,WindowLength,SequenceLength);
                        TestLabels{FileIndex} = BinaryLabels(1:TestFrameStep:end)+1;
                        ResultPathsTest{FileIndex} = fullfile(ResultsFolder,TestFileName);
                        TestFilesAnnotationDARPA{FileIndex} = regexprep(TestFilesAnnotation{FileIndex}, 'txt_S', 'txt');
                        
%                         spec = load(fullfile(SpecGramDir,TestFileName));
%                         subplot(3,1,1); imagesc(log(spec.Spectrogram(5:50,1:1000)));
%                         subplot(3,1,2); imagesc(TestData{FileIndex}(:,1:100));
%                         subplot(3,1,3); plot(TestLabels{FileIndex}(1:1000));
                    end
                    fprintf('Train and test data collected in %fsec.\n',toc);
                    
                    % create model(s) paths
                    if strcmp(Classifier(1:3),'knn')
                        ModelFilePath = fullfile(ModelFilesDir,sprintf('VadModel_%s_neighbors%d_%s',Classifier,max(Parameter),'rbfeat'));
                    else
                        ModelFilePath = cell(1,length(Parameter));
                        for ModelsIndx = 1:length(Parameter)
                            ModelFilePath{ModelsIndx} = fullfile(ModelFilesDir,sprintf('VadModel_%s_mix%d_%s',Classifier,Parameter(ModelsIndx),'rbfeat'));
                        end
                    end
                    
                    % Classification
                    tic
                    [ClassProbabilities] = Classify(TrainData,TestData,TrainLabels,TestLabels,Classifier,Parameter,'/tmp/',ModelFilePath);
                    fprintf('Classifier run in %fsec.\n',toc);
                    
                    % compute class decision
                    [ClassDecision] = ConvertClassProbabilitiesToClassDecision(ClassProbabilities, MedianFilterLength);
                    
                    % compute success percentage
                    FramePercentage = ...
                        sum(cell2mat(ClassDecision')==repmat(cell2mat(TestLabels)',1,length(Parameter)))/length(cell2mat(TestLabels));
                    
                    % convert class decision and labels into 0/1s (instead of 1/2s) and
                    % write results to predefined files for best neighbor
                    [maxval OptimalParameterPos] = max(FramePercentage);
                    format = '%d\n';
                    for i = 1:length(ClassDecision)
                        ClassDecision{i} = ClassDecision{i}-1;
                        TestLabels{i} = TestLabels{i}-1;
                        fid = fopen(ResultPathsTest{i},'w');
                        fprintf(fid,format,ClassDecision{i}(:,OptimalParameterPos)');
                        fclose(fid);
                        
% % %                         [TestFileDir TestFileName] = fileparts(TestFilesAudio{i});
% % %                         spec = load(fullfile(SpecGramDir,TestFileName));
% % %                         subplot(4,1,1); imagesc(log(spec.Spectrogram(5:50,1:1000)));
% % %                         subplot(4,1,2); imagesc(TestData{i}(:,1:100));
% % %                         subplot(4,1,3); plot(TestLabels{i}(1:100));
% % %                         subplot(4,1,4); plot(ClassDecision{i}(1:100));
% % %                         pause
                    end
                    fprintf('Optimal Parameter = %d\n',Parameter(OptimalParameterPos));
                    disp(FramePercentage);
                    
                    % run DARPA evaluation script
                    TaskId = sprintf('SmallTest_%s_%s_ch%s',Classifier,'rbfeat',Channels{i});
                    fid = fopen(DARPAEvalInputFile,'w');
                    for i = 1:length(TestFilesAnnotationDARPA)
                        fprintf(fid,'%s\t%s\t%s\n',TestFilesAudio{i},TestFilesAnnotationDARPA{i},ResultPathsTest{i});
                    end
                    fclose(fid);
                    [Pmiss Pfa] = FindPercentageFromResultFiles(DARPAEvalInputFile,EvalScriptPath,AudioDir,WorkingDir,TaskId,RbFeaturesShift,WindowLength);
                    fprintf('\nPmiss = %f\t Pfa = %f\n',Pmiss,Pfa);
                    fprintf('\n**************************************************\n\n');
                    
                    % write results report file
                    FeatureParams = sprintf('Smooth%d_Window%d_AlphaX%d_AlphaY%d',MovingIndex,AnalysisWindow(AnalysisWindowIndex),100*XBandAlpha(XBandAlphaIndex),100*YBandAlpha(YBandAlphaIndex));
                    fprintf(OverallResultsFid,'%s-%d\t%s\t%d\t%f\t%f\t%f\t%f\n',DimReducTechnique,DimReducParam,FeatureParams,Parameter(OptimalParameterPos),FramePercentage(OptimalParameterPos),Pmiss,Pfa,(Pmiss+Pfa)/2);
                end
            end
        end
    end
    
    fclose(OverallResultsFid);
    
end