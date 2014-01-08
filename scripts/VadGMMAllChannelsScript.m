function VadGMMAllChannelsScript(Feature)
% runs VAD with GMM classification for each channel separately
% Feature   : the feature based on which we run K-NN (MFCCs,LTSV_R100_e0, etc)

% define K-NN classifier
Classifier = 'gmm';

% channels
Channels = {'A' 'B' 'C' 'D' 'F' 'G' 'H' 'src'};
Channels = {'C'};

% number of mixtures to test
Mixtures = [2 4];

% length of median filtering to smooth final labels
MedianFilterLength = 51;

% directory of 4-column input files
InputFilesDir = '/home/theodora/RATS/scripts/FilenamePaths/';

% directory to save models
ModelsDir = '/home/theodora/RATS/models/VadModels/';

% different input files for different features
if strcmp(Feature,'MFCCs');
    
    % name of variable in which the features have been save in .mat format
    FeaturesStructName = 'Cepstra';
    % middle name of 4-column input file
    InputFileMiddleName = 'FileWavResFeatpaths';
   
elseif strcmp(Feature(1:4),'LTSV')
    % name of variable in which the features have been save in .mat format
    FeaturesStructName = 'LTSV';
    % middle name of 4-column input file
    Feature(Feature=='_')=[];
    InputFileMiddleName = sprintf('FileWavRes%sFeatpaths',Feature);
end

% classify with GMMs
for i = 1:length(Channels)
    fprintf('**************************************************\nChannel %s\n',Channels{i});
    TrainFilesPaths = fullfile(InputFilesDir,sprintf('%s_%s_Train.txt',Channels{i},InputFileMiddleName));
    TestFilesPaths = fullfile(InputFilesDir,sprintf('%s_%s_Test.txt',Channels{i},InputFileMiddleName));
    ModelFilePath = cell(1,length(Mixtures));
    for j = 1:length(Mixtures)
        ModelFilePath{j} = fullfile(ModelsDir,sprintf('VadModel_GMMs_mix%d_%s_ch%s',Mixtures(j),Feature,Channels{i}));
    end
    TaskId = sprintf('Test_GMM_%s_ch%s',Feature,Channels{i});
    [Pmiss Pfa FramePercentage] = VadClassify(TrainFilesPaths,TestFilesPaths,FeaturesStructName,ModelFilePath,Classifier,Mixtures,MedianFilterLength,TaskId);
    disp(FramePercentage);
    fprintf('\nPmiss = %f\t Pfa = %f\n',Pmiss,Pfa);
    fprintf('\n**************************************************\n\n');
end