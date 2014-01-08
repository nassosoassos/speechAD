%  function VadGMMs()

% boolean whether or not to train
TrainBoolean = 1;

% features used for channel estimation
% Features = 'LTSV_R500_e0';
Features = 'MFCCs';
FeaturesStructName = 'Cepstra';

% features input directory
InputDir = fullfile('/media/DATTRANSFER/RATS/Features',Features);

% labels input directory
LabelsDir = fullfile('../LDC2011E87/LDC2011E87_First_Incremental_SAD_Annotation/data/train');

% models output directory
ModelsOutputDir = fullfile('../models/VadModels');

% results output directory
ResultsOutputDir = fullfile('../results',strcat('GMMs_',Features));

% channels
Channels = ['A','B','C','D','E','F','G','H','S'];

% number of Gaussians for each channel
K = 16;

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

% length of median filtering from final decision (in frames)
MedianFilterLength = 51;

if TrainBoolean
    
    for i = 1:length(Channels)
        % collect all data for training
        tic
        TrainData = [];
        TrainLabels = [];
        for j = 1:20%length(TrainFilesDir{1})
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
                if strcmp(Features,'MFCCs')
                    SequenceLength = size(AudioFeatures.(FeaturesStructName),2);
                else
                    SequenceLength = size(AudioFeatures,2);
                end
                SpeechSegments = load(fullfile(LabelsDir,LanguageName,'sad',ChannelName,strcat(FileName,'.txt_S')));
                BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
                if strcmp(Features,'MFCCs')
                    TrainData = [TrainData AudioFeatures.(FeaturesStructName)(:,1:TrainFrameStep:end)];
                else
                    TrainData = [TrainData AudioFeatures(:,1:TrainFrameStep:end)];
                end
                TrainLabels = [TrainLabels BinaryLabels(1:TrainFrameStep:end)];
            end
        end
        TrainDataSpeech = TrainData(:,TrainLabels==1);
        TrainDataNonSpeech = TrainData(:,TrainLabels==0);
        fprintf('Channel %c:\t data collected\t frames=%d\t %fsec\n',Channels(i),size(TrainData,2), toc);
        clear TrainData TrainLabels;
        tic
        % create GMM object
        MixSpeech = gmm(size(TrainDataSpeech,1), K, 'full');
        MixNonSpeech = gmm(size(TrainDataNonSpeech,1), K, 'full');
        % initialize GMMs
        options = foptions;
        options(14) = 50;	% Just use 5 iterations of k-means in initialisation
        MixSpeech = gmminit(MixSpeech, TrainDataSpeech', options);
        MixNonSpeech = gmminit(MixNonSpeech, TrainDataNonSpeech', options);
        % run EM
        options = zeros(1, 18);
        options(1)  = 1;		% Prints out error values.
        options(14) = 50;		% Number of iterations.
        [MixSpeech, options, errlog] = gmmem_htk(MixSpeech, TrainDataSpeech', options);
        [MixNonSpeech, options, errlog] = gmmem_htk(MixNonSpeech, TrainDataNonSpeech', options);
        % save model
        if strcmp(ChannelName,'src')
            save(fullfile(ModelsOutputDir,strcat('VadModel_GMM_mix',int2str(K),'_',Features,'_src')),'MixSpeech','MixNonSpeech');
        else
            save(fullfile(ModelsOutputDir,strcat('VadModel_GMM_mix',int2str(K),'_',Features,'_ch',Channels(i))),'MixSpeech','MixNonSpeech');
        end
        fprintf('Channel %c:\t GMM trained\t %fsec\n',Channels(i),toc);
        %     ChannelModels{i} = gmdistribution.fit(TrainData',K);
        clear TrainDataSpeech TrainDataNonSpeech;
    end
    
end

% load channel models
ChannelModels = cell(2,length(Channels));
for i = 1:length(Channels)
    y = load(fullfile(ModelsOutputDir,strcat('VadModel_GMM_mix',int2str(K),'_',Features,'_ch',Channels(i))),'MixSpeech','MixNonSpeech');
    ChannelModels{1,i} = y.MixSpeech; ChannelModels{2,i} = y.MixNonSpeech;
end

% test GMMs based on sum of maximum log-likelihoods
CorrectFramesPerChannel = zeros(1,length(Channels));
TotalFramesPerChannel = zeros(1,length(Channels));
for i = 1:length(TestFilesDir{1})
    FileName = TestFilesDir{1}{i};
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
    if strcmp(Features,'MFCCs')
        SequenceLength = size(AudioFeatures.(FeaturesStructName),2);
    else
    end
    SpeechSegments = load(fullfile(LabelsDir,LanguageName,'sad',ChannelName,strcat(FileName,'.txt_S')));
    BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
    if strcmp(Features,'MFCCs')
        Likelihood = zeros(size(AudioFeatures.(FeaturesStructName),2),2);
        Likelihood(:,1) = gmmprob(ChannelModels{1,ChannelIndx}, AudioFeatures.(FeaturesStructName)');
        Likelihood(:,2) = gmmprob(ChannelModels{2,ChannelIndx}, AudioFeatures.(FeaturesStructName)');
    else
        Likelihood = zeros(size(AudioFeatures,2),2);
        Likelihood(:,1) = gmmprob(ChannelModels{1,ChannelIndx}, AudioFeatures');
        Likelihood(:,2) = gmmprob(ChannelModels{1,ChannelIndx}, AudioFeatures');
    end
    [MaxLik TestDecision] = max(Likelihood,[],2);
    TestDecision(TestDecision==2)=0;
    TestDecision = round(medfilt1(TestDecision,MedianFilterLength));
    TotalFramesPerChannel(ChannelIndx) = TotalFramesPerChannel(ChannelIndx)+SequenceLength;
    CorrectFramesPerChannel(ChannelIndx) = CorrectFramesPerChannel(ChannelIndx)+sum(TestDecision'==BinaryLabels);
    fprintf('Test File %s\t Length %d\t Correct %d\t Percentage %f\n',FileName,SequenceLength,sum(TestDecision'==BinaryLabels),sum(TestDecision'==BinaryLabels)/SequenceLength);
    disp(CorrectFramesPerChannel./TotalFramesPerChannel);
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