 function Vad(Classifier,K)
 
% VAD based on Features using GMM or KNN classifier
% Classifier: string containg the name of the classifier
%             "GMMs" for GMM classifier
%             "KNN" for K-NN classifier
% K         : number of gaussians for GMM classifier
%           : number of nearest neighbors for K-NN classifier

% boolean whether or not to train
TrainBoolean = 1;

% features used for channel estimation
Features = 'MFCCs';
FeaturesStructName = 'Cepstra';

% features input directory
InputDir = fullfile('/media/DATTRANSFER/RATS/Features',Features);

% labels input directory
LabelsDir = fullfile('../LDC2011E87/LDC2011E87_First_Incremental_SAD_Annotation/data');

% models output directory
ModelsOutputDir = fullfile('../models/VadModels');

% results output directory
ResultsOutputDir = fullfile('/home/theodora/RATS/results',strcat(Classifier,'_',Features));

% channels
Channels = ['A','B','C','D','E','F','G','H','S'];

% frame step for including frames in training
TrainFrameStep = 10;

% length of median filter
MedianFilterLength = 50;

% files containing paths of train and test audio samples
TrainFilesFid = fopen('./TrainFiles');
TestFilesFid = fopen('./TestFiles');

TrainFilesDir = textscan(TrainFilesFid, '%s\n');
TestFilesDir = textscan(TestFilesFid, '%s\n');

% feature extraction window and step parameters
Shift = 0.01;
WindowLength = 0.02;

CorrectFramesPerChannel = zeros(1,length(Channels));
TotalFramesPerChannel = zeros(1,length(Channels));

if TrainBoolean
    
    for i = 1:length(Channels)
        % collect all data for training
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
                elseif k==DirectoryCount-3
                    ModeName = rest;    % train or dev-1
                end
            end
            FileName = FileName(2:end-4);
            if ChannelIndx==i
                AudioFeatures = load(fullfile(InputDir,strcat(FileName,'.mat')));
                SequenceLength = size(AudioFeatures.(FeaturesStructName),2);
                SpeechSegments = load(fullfile(LabelsDir,ModeName,LanguageName,'sad',ChannelName,strcat(FileName,'.txt_S')));
                BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
                TrainData = [TrainData AudioFeatures.(FeaturesStructName)(:,1:TrainFrameStep:end)];
                TrainLabels = [TrainLabels BinaryLabels(1:TrainFrameStep:end)];
            end
        end
        fprintf('Channel %c:\t training data collected\t frames=%d\t %fsec\n',Channels(i),size(TrainData,2), toc);
        if strcmp(Classifier,'GMMs')
            % separate speech and non-speech samples
            TrainDataSpeech = TrainData(:,TrainLabels==1);
            TrainDataNonSpeech = TrainData(:,TrainLabels==0);
            clear TrainData TrainLabels;
            tic
            % create GMM object
            MixSpeech = gmm(size(TrainDataSpeech,1), K, 'full');
            MixNonSpeech = gmm(size(TrainDataNonSpeech,1), K, 'full');
            % initialize GMMs
            options = foptions;
            options(14) = 50;	% Just use 5 iterations of k-means in initialization
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
                save(fullfile(ModelsOutputDir,strcat('VadModel_',Classifier,'_mix',int2str(K),'_',Features,'_src')),'MixSpeech','MixNonSpeech');
            else
                save(fullfile(ModelsOutputDir,strcat('VadModel_',Classifier,'_mix',int2str(K),'_',Features,'_ch',Channels(i))),'MixSpeech','MixNonSpeech');
            end
            fprintf('Channel %c:\t %s trained\t %fsec\n',Classifier,Channels(i),toc);
            clear TrainDataSpeech TrainDataNonSpeech;
        end
        if strcmp(Classifier,'KNN')
            % collect data for testing
            tic
            TestData = [];
            TestLabels = [];
            AccumSamplesCounter = [1];
            IndicesPerChannel = [];
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
                    SpeechSegments = load(fullfile(LabelsDir,ModeName,LanguageName,'sad',ChannelName,strcat(FileName,'.txt_S')));
                    BinaryLabels = ConvertTimeStampsToFrames(SpeechSegments,Shift,WindowLength,SequenceLength);
                    TestData = [TestData AudioFeatures.(FeaturesStructName)];
                    TestLabels = [TestLabels BinaryLabels];
                    AccumSamplesCounter(end+1) = AccumSamplesCounter(end)+SequenceLength;
                    IndicesPerChannel = [IndicesPerChannel j];
                end
            end
            if isempty(TestData)
                continue
            end
            fprintf('Channel %c:\t testing data collected\t frames=%d\t %fsec\n',Channels(i),size(TestData,2), toc);
            % write test and train data into a temporary file
            format = repmat('%f ',1,size(TrainData,1)); format(end:end+1) = '\n';
            fid = fopen('/tmp/train','w');
            fprintf(fid,format,TrainData);
            fclose(fid);
            fid = fopen('/tmp/test','w');
            fprintf(fid,format,TestData);
            fclose(fid);
            % do K-NN classification for a specific channel
            cmd = sprintf('ann_sample -d %d -nn %d -max %d -df %s -qf %s | grep -v "(" | grep -v "NN" | grep -v Data > %s%s',size(TrainData,1),K,size(TrainData,2),'/tmp/train','/tmp/test','/tmp/knn_result',Channels(i));
            disp(cmd);
            system(cmd);
            fprintf('K-NN finished in %fsec\n',toc);
            Results = load(strcat('/tmp/knn_result',Channels(i)));
            NearestNeighbor = buffer(Results(:,2),K);
            NearestNeighborDistance = buffer(Results(:,3),K);
            % make final decision for each sample
            for j = 1:length(IndicesPerChannel)
                FileName = TestFilesDir{1}{IndicesPerChannel(j)};
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
                BinaryLabels = TestLabels(AccumSamplesCounter(j):AccumSamplesCounter(j+1)-1);
                for kk = 31:31
                    TestDecision = zeros(1,AccumSamplesCounter(j+1)-AccumSamplesCounter(j));
                    for k = AccumSamplesCounter(j):AccumSamplesCounter(j+1)-1
                        NumberOfSpeechNeighbors = sum(TrainLabels(NearestNeighbor(1:kk,k)+1)==1);
                        if NumberOfSpeechNeighbors>kk/2
                            TestDecision(k-AccumSamplesCounter(j)+1) = 1;
                        end
                    end
                    %                     NumberOfNonSpeechNeighbors = length(TrainLabels(NearestNeighbor(:,k)+1)==0);
                    TestDecision = round(medfilt1(TestDecision,MedianFilterLength));
                    fprintf('%d Test File %s\n Length %d\t Correct %d\t Percentage %f\n',kk,FileName,AccumSamplesCounter(j+1)-AccumSamplesCounter(j),sum(TestDecision==BinaryLabels),sum(TestDecision==BinaryLabels)/length(TestDecision));
                end
                TotalFramesPerChannel(ChannelIndx) = TotalFramesPerChannel(ChannelIndx)+SequenceLength;
                CorrectFramesPerChannel(ChannelIndx) = CorrectFramesPerChannel(ChannelIndx)+sum(TestDecision==BinaryLabels);
                disp(CorrectFramesPerChannel(i)./TotalFramesPerChannel(i));
                OutputDir = fullfile(ResultsOutputDir,LanguageName,ChannelName);
                if ~exist(OutputDir,'dir')
                    mkdir(OutputDir);
                end
                fid = fopen(fullfile(OutputDir,FileName),'w');
                fprintf(fid,'%d\n',TestDecision);
                fclose(fid);
            end
        end
    end
    
end


if strcmp(Classifier,'KNN')
    FramePercentagePerChannel = CorrectFramesPerChannel./TotalFramesPerChannel;
    for i = 1:length(Channels)
        fprintf('Channel %d:\t total:%d\t\t correct:%d\t\t percentage:%f\n',i,TotalFramesPerChannel(i),CorrectFramesPerChannel(i),FramePercentagePerChannel(i));
    end
end

% load channel models
if strcmp(Classifier,'GMMs')
    ChannelModels = cell(2,length(Channels));
    for i = 1:length(Channels)-1
        if strcmp(Channels(i),'S')
            y = load(fullfile(ModelsOutputDir,strcat('VadModel_',Classifier,'_mix',int2str(K),'_',Features,'_src')),'MixSpeech','MixNonSpeech');
        else
            y = load(fullfile(ModelsOutputDir,strcat('VadModel_',Classifier,'_mix',int2str(K),'_',Features,'_ch',Channels(i))),'MixSpeech','MixNonSpeech');
        end
        ChannelModels{1,i} = y.MixSpeech; ChannelModels{2,i} = y.MixNonSpeech;
    end
end

% test GMMs based on sum of maximum log-likelihoods
if strcmp(Classifier,'GMMs')
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
        Likelihood(:,2) = gmmprob(ChannelModels{2,ChannelIndx}, AudioFeatures.(FeaturesStructName)');
        [MaxLik TestDecision] = max(Likelihood,[],2);
        TestDecision(TestDecision==2)=0;
        TestDecision = round(medfilt1(TestDecision,MedianFilterLength));
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
end


AnnotDir = '/home/theodora/RATS/LDC2011E87/LDC2011E87_First_Incremental_SAD_Annotation/data/train';
ScoringEnginePath = '/home/theodora/RATS/RES_v1-2/RES_1-2_ScoringEngine.jar';
AudioDir = '/home/theodora/RATS/sad_ldc2011e86_v2/data/train';
for i = 1:length(Channels)
    
    fid = fopen('/tmp/result_directories','w');
    
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
            OutputDir = fullfile(ResultsOutputDir,LanguageName,ChannelName);
            fprintf(fid,'%s\n',fullfile(OutputDir,FileName));
        end
    end
    
    fclose(fid);
    
    [Pmiss Pfa] = FindPercentageFromResultFiles('/tmp/result_directories',AnnotDir,ScoringEnginePath,AudioDir);
    
    fprintf('Channel %d:\t Pmiss:%f\t Pfa:%f\t Average:%f\n',i,Pmiss, Pfa, 0.5*(Pfa+Pmiss));
    
end


fclose(TrainFilesFid);
fclose(TestFilesFid);