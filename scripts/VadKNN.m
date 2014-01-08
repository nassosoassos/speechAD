%  function VadKNN()

% features used for channel estimation
Features = 'MFCCs';
FeaturesStructName = 'Cepstra';

% features input directory
InputDir = fullfile('/media/DATTRANSFER/RATS/Features',Features);

% labels input directory
LabelsDir = fullfile('../LDC2011E87/LDC2011E87_First_Incremental_SAD_Annotation/data/train');

% models output directory
ModelsOutputDir = fullfile('../models/VadModels');

% results output directory
ResultsOutputDir = fullfile('../results',strcat('KNN_',Features));

% channels
Channels = ['A','B','C','D','E','F','G','H','S'];

% frame step for including frames in training
TrainFrameStep = 10;

% files containing paths of train and test audio samples
TrainFilesFid = fopen('./TrainFiles');
TestFilesFid = fopen('./TestFiles');

TrainFilesDir = textscan(TrainFilesFid, '%s\n');
TestFilesDir = textscan(TestFilesFid, '%s\n');

% feature extraction window and step parameters
Shift = 0.01;
WindowLength = 0.02;

for i = 1:length(Channels)
    % collect data for training
    tic
    TrainData = [];
    TrainLabels = [];
    for j = 1:length(TrainFilesDir{1})
        FileName = TrainFilesDir{1}{j};
        DirectoryCount = sum(ismember(FileName,'/'))-1;
        for k = 1:DirectoryCount
            [rest FileName] = strtok(FileName,'/');
            if k==DirectoryCount
                ChannelName = rest(1:end-6);
                ChannelIndx = find(upper(ChannelName(1))==Channels);
            elseif k==DirectoryCount-2
                LanguageName = rest;
            end
        end
        FileName = FileName(2:end-4);
        if ChannelIndx==i
            AudioFeatures = load(fullfile(InputDir,strcat(FileName,'.mat')));
            SequenceLength = size(AudioFeatures.(FeaturesStructName),2);
            SpeechSegments = load(fullfile(LabelsDir,LanguageName,'sad',ChannelName,strcat(FileName,'.txt_S')));
            BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
            TrainData = [TrainData AudioFeatures.(FeaturesStructName)(:,1:TrainFrameStep:end)];
            TrainLabels = [TrainLabels BinaryLabels(1:TrainFrameStep:end)];
        end
    end
    fprintf('Channel %c:\t training data collected\t frames=%d\t %fsec\n',Channels(i),size(TrainData,2), toc);
    
    % collect data for testing
    tic
    TestData = [];
    TestLabels = [];
    for j = 1:length(TestFilesDir{1})
        FileName = TestFilesDir{1}{j};
        DirectoryCount = sum(ismember(FileName,'/'))-1;
        for k = 1:DirectoryCount
            [rest FileName] = strtok(FileName,'/');
            if k==DirectoryCount
                ChannelName = rest(1:end-6);
                ChannelIndx = find(upper(ChannelName(1))==Channels);
            elseif k==DirectoryCount-2
                LanguageName = rest;
            end
        end
        FileName = FileName(2:end-4);
        if ChannelIndx==i
            AudioFeatures = load(fullfile(InputDir,strcat(FileName,'.mat')));
            SequenceLength = size(AudioFeatures.(FeaturesStructName),2);
            SpeechSegments = load(fullfile(LabelsDir,LanguageName,'sad',ChannelName,strcat(FileName,'.txt_S')));
            BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
            TestData = [TestData AudioFeatures.(FeaturesStructName)];
            TestLabels = [TestLabels BinaryLabels];
        end
    end
    fprintf('Channel %c:\t testing data collected\t frames=%d\t %fsec\n',Channels(i),size(TestData,2), toc);
    
    tic
    % K-NN classification
    
    fprintf('Channel %c:\t K-NN\t %fsec\n',Channels(i),toc);
    %     ChannelModels{i} = gmdistribution.fit(TrainData',K);
    clear TrainData TestData;
end


% test GMMs based on sum of maximum log-likelihoods
CorrectFramesPerChannel = zeros(1,length(Channels));
TotalFramesPerChannel = zeros(1,length(Channels));
for i = 1:length(TestFilesDir{1})
    FileName = TrainFilesDir{1}{j};
    DirectoryCount = sum(ismember(FileName,'/'))-1;
    for k = 1:DirectoryCount
        [rest FileName] = strtok(FileName,'/');
        if k==DirectoryCount
            ChannelName = rest(1:end-6);
            ChannelIndx = find(upper(ChannelName(1))==Channels);
        elseif k==DirectoryCount-2
            LanguageName = rest;
        end
    end
    FileName = FileName(2:end-4);
    AudioFeatures = load(fullfile(InputDir,strcat(FileName,'.mat')));
    SequenceLength = size(AudioFeatures.(FeaturesStructName),2);
    SpeechSegments = load(fullfile(LabelsDir,LanguageName,'sad',ChannelName,strcat(FileName,'.txt_S')));
    BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
    Likelihood = zeros(size(AudioFeatures.(FeaturesStructName),2),2);
    Likelihood(:,1) = gmmprob(ChannelModels{1,ChannelIndx}, AudioFeatures.(FeaturesStructName)');
    Likelihood(:,2) = gmmprob(ChannelModels{1,ChannelIndx}, AudioFeatures.(FeaturesStructName)');
    [MaxLik TestDecision] = max(Likelihood,[],2);
    TotalFramesPerChannel(ChannelIndx) = TotalFramesPerChannel(ChannelIndx)+SequenceLength;
    CorrectFramesPerChannel(ChannelIndx) = CorrectFramesPerChannel(ChannelIndx)+sum(TestDecision'==BinaryLabels);
    fprintf('Test File %s\t Length %d\t Correct %d\t Percentage %f\n',FileName,SequenceLength,sum(TestDecision'==BinaryLabels),sum(TestDecision'==BinaryLabels)/SequenceLength);
    OutputDir = fullfile(ResultsOutputDir,LanguageName,ChannelName);
    if ~exist(OutputDir,'dir')
        mkdir(OutputDir);
    end
    fid = fopen(fullfile(OutputDir,FileName),'w');
    fprintf(fid,'%d\n',TestDecision');
    fclose(fid);
end
FramePercentagePerChannel = CorrectFramesPerChannel./TotalFramesPerChannel;

for i = 1:length(Channels)
    fprintf('Channel %d:\t total:%d\t\t correct:%d\t\t percentage:%f\n',i,TotalFramesPerChannel(i),CorrectFramesPerChannel(i),FramePercentagePerChannel(i));
end

fclose(TrainFilesFid);
fclose(TestFilesFid);