function mmf_struct=msMmf2masv(filename)
% Reads the params of the htk-model found in filename into an appropriate
% structure mmf_struct. The htk model can be multi-stream
  
% mmf_struct = mmf2matlab(filename)
%
% CVS_Version_String = '$Id: mmf2matlab.m,v 1.4 2003/12/10 20:00:46 tuerk Exp $';
% CVS_Name_String = '$Name: rel-1-4-01 $';

% ###########################################################
%
% This file is part of the matlab scripts of the MASV System.
% MASV = Munich Automatic Speaker Verification
%
% Copyright 2002-2003, Ulrich T�rk
% Institute of Phonetics and Speech Communication
% University of Munich
% tuerk@phonetik.uni-muenchen.de

if nargin < 1
  filename = 'ms_zero';
end
  
covkind_cell = {'DIAGC','INVDIAGC','FULLC','LLTC','XFORMC'};
durkind_cell = {'NULLD','POISSOND','GAMMAD','GEND'};
parm_kind_base_cell = {'DISCRETE','LPC','LPCEPSTRA','MFCC','FBANK','MELSPEC','LPREFC','LPDELCEP','USER','MFCC_D', 'MFCC_D_Z', 'MFCC_D_A', 'MFCC_D_A_Z'};

line_array=textread(filename,'%s','delimiter','\n','whitespace','');

total_lines = length(line_array);

% first find the number of streams
numStreams=1;  % default
for line = 1:total_lines,  
  tmp_line = line_array{line};  
  if (findstr('<STREAMINFO>', tmp_line)),  % eg <STREAMINFO> 2 39 45
    close_par = findstr('>',tmp_line);
    streamInfoVec = sscanf(tmp_line(close_par(1)+1:end), '%d');
    numStreams = streamInfoVec(1);
    for i=1:numStreams
      mmf_struct(i).vector_size=streamInfoVec(i+1);
    end
    break;
  end
end

for i=1:numStreams
  mmf_struct(i).name=filename; 
  mmf_struct(i).state=[];
end
  
state_counter = 0;
if numStreams>1, 
  stream_counter = 0; 
else 
  stream_counter=1;
end
mix_counter = 0;

for line = 1:total_lines,  

  tmp_line = line_array{line};
  
  if (findstr('~h', tmp_line)),  % e.g. ~h "zero"
    quotes = findstr('"',tmp_line);
    hmm_name = sscanf(tmp_line(quotes(1)+1:quotes(2)-1),'%s');
    for i=1:numStreams
      mmf_struct(i).name = hmm_name;
    end
  end

  if (findstr('<VECSIZE>', tmp_line)),  % e.g. <VECSIZE> 84<NULLD>
    open_par = findstr('<',tmp_line);
    close_par = findstr('>',tmp_line);
    if numStreams==1
      mmf_struct(1).vector_size = sscanf(tmp_line(close_par(1)+1:open_par(2)-1),'%d');
    end
    for i=2:length(open_par)
      keyword = tmp_line(open_par(i) + 1:close_par(i)-1);
      if(findstr_inCell(keyword,covkind_cell)),
        for i=1:numStreams
          mmf_struct(i).covKind = keyword;
        end
      end
      if(findstr_inCell(keyword,durkind_cell)),
        for i=1:numStreams
          mmf_struct(i).durKind = keyword;
        end
      end
      if(findstr_inCell(keyword,parm_kind_base_cell)),
        for i=1:numStreams
          mmf_struct(i).vector_type = keyword;
        end
      end
    end
  end
  
  if (findstr('<NUMSTATES>',tmp_line)),
    for i=1:numStreams
      mmf_struct(i).num_of_states = sscanf(tmp_line(length('<NUMSTATES>')+1:end),'%d');
    end
  end
  
  if (findstr('<STATE>',tmp_line)),
    state_counter = state_counter + 1;
    mix_counter = 0;
    if numStreams>1, 
      stream_counter = 0; 
    end
    for i=1:numStreams
      mmf_struct(i).state(state_counter).mix_number = 1;
      mmf_struct(i).state(state_counter).state_number = sscanf(tmp_line(length('<STATE>')+1:end),'%d');
    end
  end
  
  if (findstr('<NUMMIXES>',tmp_line)),
    number_of_mixtures_at_state = sscanf(tmp_line(length('<NUMMIXES>')+1:end),'%d');
    for i=1:numStreams
      mmf_struct(i).state(state_counter).mix_number = number_of_mixtures_at_state;
      for m_counter=1:number_of_mixtures_at_state,
        mmf_struct(i).state(state_counter).mix_weight(m_counter) = 0;
      end
    end
  end
  
  if (findstr('<SWEIGHTS>',tmp_line)),   % <SWEIGHTS> 2 \n 2.000000e-01 1.000000e+00
    tmp_line_next = line_array{(line+1)};
    sWeightVec = sscanf(tmp_line_next, '%f');
    for i=1:numStreams
      mmf_struct(i).state(state_counter).stream_weight = sWeightVec(i);
    end
  end
  
  if (findstr('<STREAM>',tmp_line)),
    stream_counter = stream_counter + 1;
    mix_counter = 0;
  end
  
  if (findstr('<MIXTURE>',tmp_line)),
    temp= sscanf(tmp_line(length('<MIXTURE>')+1:end),'%d%f');
    mix_counter = temp(1);
    mmf_struct(stream_counter).state(state_counter).mix_weight(mix_counter) = temp(2);
  end
  
  if (findstr('<MEAN>',tmp_line)),
    line = line+1;
    tmp_line_next = line_array{line};
    if (mix_counter == 0)
      mix_counter = 1;
    end
    mmf_struct(stream_counter).state(state_counter).mean_vector(mix_counter,:) = sscanf(tmp_line_next,'%f');
  end
  
  if (findstr('<VARIANCE>',tmp_line)),
    line = line+1;
    tmp_line_next = line_array{line};
    mmf_struct(stream_counter).state(state_counter).variance_vector(mix_counter,:) = sscanf(tmp_line_next,'%f');
  end
  
  if (findstr('<INVCOVAR>',tmp_line)),
    N = sscanf(tmp_line(length('<INVCOVAR>')+1:end),'%d');
    for i=1:N,
      line = line+1;
      tmp_line_next = line_array{line};
      mmf_struct(stream_counter).state(state_counter).invcovar(mix_counter,i,:) = [ zeros(i-1,1); sscanf(tmp_line_next,'%f')   ]';
    end
  end
  
  if (findstr('<GCONST>',tmp_line)),
    mmf_struct(stream_counter).state(state_counter).gconst(mix_counter,:) = sscanf(tmp_line(length('<GCONST>')+1:end),'%f');
  end
  
  if (findstr('<TRANSP>',tmp_line)), % assume shared between streams
    mmf_struct(1).A_size = sscanf(tmp_line(length('<TRANSP>')+1:end),'%d');
    for i = 1:mmf_struct(1).A_size,
      line = line+1;
      tmp_line_next = line_array{line};
      tmp_vector = sscanf(tmp_line_next,'%f');
      mmf_struct(1).A(i,:) = tmp_vector;
    end
    for s=2:numStreams
      mmf_struct(s).A_size = mmf_struct(1).A_size;
      mmf_struct(stream_counter).A = mmf_struct(1).A;
    end
  end
  
end % for line

for s=1:numStreams
  for i=1:length(mmf_struct(s).state),
    if (~(isfield(mmf_struct(s).state(i),'mix_weight'))),
      mmf_struct(s).state(i).mix_weight(1) = 1;
    end
  end
end

for s=1:numStreams
  for state_counter=1:length(mmf_struct(s).state),	
    for m_counter=1:mmf_struct(s).state(state_counter).mix_number
      if ( size(mmf_struct(s).state(state_counter).mix_weight) == 0 ) & ( mmf_struct(s).state(state_counter).mix_number == 1 )
        mmf_struct(s).state(state_counter).mix_weight = 1;
      end
      if (mmf_struct(s).state(state_counter).mix_weight(m_counter) == 0),
        mmf_struct(s).state(state_counter).mean_vector(m_counter,:) = zeros(size(mmf_struct(s).state(state_counter).mean_vector(1,:)));
        mmf_struct(s).state(state_counter).variance_vector(m_counter,:) = zeros(size(mmf_struct(s).state(state_counter).variance_vector(1,:)));
        mmf_struct(s).state(state_counter).gconst(m_counter,:) = zeros(size(mmf_struct(s).state(state_counter).gconst(1,:)));
      end
    end
  end
end


function found = findstr_inCell(keyword,string_cell)
% FINDSTR_INCELL - 
found = false;
for i=1:length(string_cell)
  if strcmp(keyword, string_cell{i}) == 1
    found = true;
    break;
  end
end
