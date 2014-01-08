function [Pmiss Pfa] = FindPercentageFromResultFiles(InputFile,EvalScriptPath,AudioDir,WorkingDir,TaskId,Shift,WindowLength)
% finds the success percentage for all given ResutlsFiles using the
% RASTA java script
% InputFile     : a file with 3 columns for each test file, the first
%                 corresponds to the audio paths, the second to the
%                 annotation paths and the third to the result paths.
%                 The annotation files are the ones given by DARPA.
%                 The result files are a sequence of frames and corresponding
%                 0/1 (non speech,speech) decisions.
% EvalScriptPath: the path of the RATS evaluation script
% AudioDir      : the directory of all audio files (we just need to define sth)
% WorkingDir    : the directory where the program outputs the xml, txt,
%                 config files
% TaskId        : the id of the task for DARPA script
% Shift         : time shift in seconds
% WindowLength  : window length in seconds
% To run this script we need two functions:
% 1) ConvertFramesToTimeStamps: converts 0/1 results to time stamps
%                               (it is hypothesized to be in the same directory
%                               of this script or in the path)
% 2) RES_v1-2_ScoringEngine.jar : RASTA evaluation script
%                                    (its path is constant, defined by the
%                                    variable EvalScriptPath)
%  [Pmiss Pfa] = FindPercentageFromResultFiles('/tmp/B.list','/home/theodora/RATS/LDC2011E87/LDC2011E87_First_Incremental_SAD_Annotation/data/train','/home/theodora/RATS/RES_v1-2/RES_1-2_ScoringEngine.jar','/home/theodora/RATS/sad_ldc2011e86_v2/data/train');

% open input file
InputFileId = fopen(InputFile);
InputFileFormat = '%s\t%s\t%s\n';
[AllDirectories] = textscan(InputFileId,InputFileFormat);
AudioDirectories = AllDirectories{1};
AnnotationDirectories = AllDirectories{2};
ResultDirectories = AllDirectories{3};
fclose(InputFileId);

% find current path
CurrentWorkspaceDir = pwd;

% configuration parameters
SpeechCollar = 200;
NonSpeechCollar = 500;
SpeechSmoothing = 300;
NonSpeechSmoothing = 700;
AnnotationSpeechSmoothing = 200;
AnnotationNonSpeechSmoothing = 500;

% Create and open Answer, XML, Configuration and Json files (Absolute paths)
XmlFile = fullfile(WorkingDir,strcat(TaskId,'.xml'));
AnswersFile = fullfile(WorkingDir,strcat(TaskId,'.txt'));
ConfigFile = fullfile(WorkingDir,strcat(TaskId,'.properties'));
JsonFile = fullfile(WorkingDir,strcat(TaskId,'.json'));

AnswersFid = fopen(AnswersFile,'w');
XmlFid = fopen(XmlFile,'w');
ConfigFid = fopen(ConfigFile,'w');
JsonFid = fopen(JsonFile,'w');

% define test id for inner use between XML and Answers file
TestId = 'Test1';

% print XML header
XmlHeader = sprintf('<RATSTestSet id="%s" audio="%s" task="SAD">\n',TaskId,AudioDir);
fprintf(XmlFid,'%s',XmlHeader);

% print first lines of configuration file
ConfigLine = 'Phase=Test';
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = 'AnnotationFiles=';
fprintf(ConfigFid,'%s',ConfigLine);

for i = 1:length(ResultDirectories)
            
            % load 0/1 (speech/non-speech) results
            FrameSequence = load(ResultDirectories{i});
            FrameSequence = FrameSequence(:,size(FrameSequence,2));
            
            % convert frame indicators to time stamps
            [TimeStamps] = ConvertFramesToTimeStamps(FrameSequence,Shift,WindowLength);
            
            % write speech intervals to answers file
            Nsamples = size(TimeStamps,1);
            AnswersLine = sprintf('%s.xml\t%s\t%s\tSAD\tSample%d\t',TaskId,TaskId,TestId,i);
            for n = 1:Nsamples
                fprintf(AnswersFid,'%s',AnswersLine);
                fprintf(AnswersFid,'%.3f\t%.3f\n', TimeStamps(n,:)');
            end
            
            % write annotation file paths to config file
            if i==length(ResultDirectories)
                fprintf(ConfigFid,'%s\n\n',AnnotationDirectories{i});
            else
                fprintf(ConfigFid,'%s, ',AnnotationDirectories{i});
            end
            
            % write test intervals to XML file
            [status AudioFileLength] = system(sprintf('cat %s | cut -f 4 | tail -n 1',AnnotationDirectories{i}));
            XmlLine = sprintf('\t<SAMPLE id="Sample%d" file="%s">\n\t\t<SEGMENT start="0.000" end="%.3f" />\n\t</SAMPLE>\n',i,AudioDirectories{i},str2double(AudioFileLength));
            fprintf(XmlFid,'%s',XmlLine);

%=======
%    
%    % load 0/1 (speech/non-speech) results
%    FrameSequence = load(ResultDirectories{1}{i});
%    FrameSequence = FrameSequence(:,2);
%    
%    % VAD parameters
%    Shift = 0.01;
%    WindowLength = 0.02;
%    
%    % convert frame indicators to time stamps
%    [TimeStamps] = ConvertFramesToTimeStamps(FrameSequence,Shift,WindowLength);
%    
%    % write speech intervals to answers file
%    Nsamples = size(TimeStamps,1);
%    AnswersLine = sprintf('%s.xml\t%s\t%s\tSAD\tSample%d\t',task,TaskId,TestId,i);
%    for n = 1:Nsamples
%        if TimeStamps(n,1)<=TimeStamps(n,2)
%            fprintf(AnswersFid,'%s',AnswersLine);
%            fprintf(AnswersFid,'%.3f\t%.3f\n', TimeStamps(n,:)');
%        else
%            msg = sprintf('Improper segment found, start time %d greater than end time %d',...
%                TimeStamps(n,1),TimeStamps(n,2));
%            warning(msg);
%        end
%    end
%    
%    % find channel and language of current file
%    FileName = ResultDirectories{1}{i};
%    [pth, bname] = fileparts(FileName);
%    filename_parts = regexp(bname,'_','split');
%    Channel = filename_parts{end};
%    if ~sum(strcmp(Channel,{'A','B','C','D','E','F','G','H','src'}))
%        error('Unexpected filename format found: %s', bname);
%    end
%    Language = filename_parts{end-1};
%    if ~sum(strcmp(Language,{'fsh-alv','fsh-eng','rats-cts-alv','rats-cts-urd'}))
%        error('Unexpected filename format found: %s', bname);
%    end
%    FileName = bname;
%    
%    % write annotation file paths to config file
%    AnnotationFilePath = fullfile(AnnotationDir,Language,'sad',Channel,strcat(FileName,'.txt'));
%    if i==length(ResultDirectories{1})
%        fprintf(ConfigFid,'%s\n\n',AnnotationFilePath);
%    else
%        fprintf(ConfigFid,'%s, ',AnnotationFilePath);
%    end
%    
%    % write test intervals to XML file
%    AudioFileName = fullfile(audio_dir,Language,'audio',strcat(Channel,'_16000'),strcat(FileName,'.wav'));
%    [status AudioFileLength] = system(sprintf('cat %s | cut -f 4 | tail -n 1',AnnotationFilePath));
%    XmlLine = sprintf('\t<SAMPLE id="Sample%d" file="%s">\n\t\t<SEGMENT start="0.000" end="%.3f" />\n\t</SAMPLE>\n',i,AudioFileName,str2double(AudioFileLength));
%    fprintf(XmlFid,'%s',XmlLine);
%    
%>>>>>>> .r5786
end

% print test sets to XML file
XmlLine = sprintf('\t<TEST id="%s">\n',TestId);
fprintf(XmlFid,'%s',XmlLine);
for j = 1:length(ResultDirectories)
    XmlLine = sprintf('\t\t<SAMPLE ref="Sample%d" />\n',j);
    fprintf(XmlFid,'%s',XmlLine);
end
XmlLine = sprintf('\t</TEST>\n');
fprintf(XmlFid,'%s',XmlLine);
fprintf(XmlFid,'%s\n','</RATSTestSet>');

% print remaining configurations in config file
ConfigLine = sprintf('TestAndResultFiles=testFiles/%s.json', TaskId);
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = 'ReportOutputFolder=output/reports';
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = sprintf('SpeechCollar=%d', SpeechCollar);
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = sprintf('NonSpeechCollar=%d', NonSpeechCollar);
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = sprintf('SpeechSmoothing=%d', SpeechSmoothing);
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = sprintf('NonSpeechSmoothing=%d', NonSpeechSmoothing);
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = sprintf('AnnotationSpeechSmoothing=%d', AnnotationSpeechSmoothing);
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = sprintf('AnnotationNonSpeechSmoothing=%d', AnnotationNonSpeechSmoothing);
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = 'InterTermApprox=false';
fprintf(ConfigFid,'%s\n\n',ConfigLine);
ConfigLine = 'OutputFormattedTime=false';
fprintf(ConfigFid,'%s\n\n',ConfigLine);

% print Answers and XML files path to json file
fprintf(JsonFid,'[\n\t{\n');
JsonLine = sprintf('\t\t"TestFile"\t"%s",\n',XmlFile);
fprintf(JsonFid,'%s',JsonLine);
JsonLine = sprintf('\t\t"ResultFile"\t"%s"\n',AnswersFile);
fprintf(JsonFid,'%s',JsonLine);
fprintf(JsonFid,'\t}\n]\n');


% close files
fclose(AnswersFid);
fclose(XmlFid);
fclose(ConfigFid);
fclose(JsonFid);


% run RATS evaluation script from evaluation directory (error otherwise)
[EvalScriptDir, EvalScriptName, EvalScriptExtension] = fileparts(EvalScriptPath);
cd(EvalScriptDir);
cmd = sprintf('java -jar %s%s -config %s -t %s -r %s',EvalScriptName, EvalScriptExtension, ConfigFile, XmlFile, AnswersFile);
disp(cmd);
[status output] = system(cmd);

% return to previous workspace directory
cd(CurrentWorkspaceDir);

% find results from report
ReportForThisTaskList = dir(fullfile(EvalScriptDir,'output','reports',strcat('Test-',TaskId,'_Report_*.txt')));
if isempty(ReportForThisTaskList)   % in case this is the first report under this task
    ReportFid = fopen(fullfile(EvalScriptDir,'output','reports',strcat('Test-',TaskId,'_Report.txt')));
else    % the most recent report is the one with the larger counter
    ReportFid = fopen(fullfile(EvalScriptDir,'output','reports',strcat('Test-',TaskId,'_Report_',num2str(length(ReportForThisTaskList)),'.txt')));
end
ReportLines = fgetl(ReportFid);
while ~strcmp(ReportLines,'Overall Results')
    ReportLines = fgetl(ReportFid);
end
for i = 1:6
    ReportLines = fgetl(ReportFid);
end
ReportLines = fgetl(ReportFid);
PmissStr = strtok(ReportLines,'Pmiss:');
ReportLines = fgetl(ReportFid);
PfaStr = strtok(ReportLines,'Pfa:');
Pmiss = str2double(PmissStr);
Pfa = str2double(PfaStr);
fclose(ReportFid);
disp(Pmiss)
disp(Pfa)
