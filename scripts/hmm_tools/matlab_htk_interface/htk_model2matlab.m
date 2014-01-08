function model_struct=htk_model2matlab(filename)
% Reads the params of the htk-model found in filename into an appropriate
% structure htkModel
% 
% 
% model_struct = htk_model2matlab( filename)
%

if nargin < 1
  filename = 'single_model_zero';
end
  
covkind_cell = {'DIAGC','INVDIAGC','FULLC','LLTC','XFORMC'};
durkind_cell = {'NULLD','POISSOND','GAMMAD','GEND'};
parm_kind_base_cell = {'DISCRETE','LPC','LPCEPSTRA','MFCC','FBANK','MELSPEC','LPREFC','LPDELCEP','USER'};

line_array=textread(filename,'%s','delimiter','\n','whitespace','');

total_lines = length(line_array);

model_struct.name=filename;
model_struct.state=[];

state_counter = 0;
mix_counter = 0;

for line = 1:total_lines,  

  tmp_line = line_array{line};  
  vec_size_pos = regexpi(tmp_line, '<VECSIZE>');
  if ~isempty(vec_size_pos),
    open_par = findstr('<',tmp_line);
    close_par = findstr('>',tmp_line);
    model_struct.vector_size = sscanf(tmp_line(close_par(1)+1:open_par(2)-1), '%d');
    for i=2:length(open_par)
      keyword = tmp_line(open_par(i) + 1:close_par(i)-1);
      if(findstr_inCell(keyword,covkind_cell)),
        model_struct.covKind = keyword;
      end
      if(findstr_inCell(keyword,durkind_cell)),
        model_struct.durKind = keyword;
      end
      if(findstr_inCell(keyword,parm_kind_base_cell)),
        model_struct.vector_type = keyword;
      end
    end
  end
  
  num_states_pos = regexpi(tmp_line, '<NUMSTATES>');
  if ~isempty(num_states_pos),
    model_struct.num_of_states = sscanf(tmp_line(length('<NUMSTATES>')+1:end),'%d');
  end
  
  state_pos = regexpi(tmp_line,'<STATE>');
  if ~isempty(state_pos),
    state_counter = state_counter + 1;
    mix_counter = 0;
    model_struct.state(state_counter).mix_number = 1;
    model_struct.state(state_counter).state_number = sscanf(tmp_line(length('<STATE>')+1:end),'%d');
  end

  nummixes_pos = regexpi(tmp_line,'<NUMMIXES>');
  if ~isempty(nummixes_pos),
    model_struct.state(state_counter).mix_number = sscanf(tmp_line(length('<NUMMIXES>')+1:end),'%d');
    for m_counter=1:model_struct.state(state_counter).mix_number,
      model_struct.state(state_counter).mix_weight(m_counter) = 0;
    end
  end
  
  mixture_pos = regexpi(tmp_line,'<MIXTURE>');
  if ~isempty(mixture_pos),
    temp= sscanf(tmp_line(length('<MIXTURE>')+1:end),'%d%f');
    mix_counter = temp(1);
    model_struct.state(state_counter).mix_weight(mix_counter) = temp(2);
  end
  
  mean_pos = regexpi(tmp_line,'<MEAN>');
  if ~isempty(mean_pos),
    line = line+1;
    tmp_line_next = line_array{line};
    if (mix_counter == 0)
      mix_counter = 1;
    end
    model_struct.state(state_counter).mean_vector(mix_counter,:) = sscanf(tmp_line_next,'%f');
  end
  
  variance_pos = regexpi(tmp_line,'<VARIANCE>');
  if ~isempty(variance_pos),
    line = line+1;
    tmp_line_next = line_array{line};
    model_struct.state(state_counter).variance_vector(mix_counter,:) = sscanf(tmp_line_next,'%f');
  end

  invcovar_pos = regexpi(tmp_line,'<INVCOVAR>');
  if ~isempty(invcovar_pos),
    N = sscanf(tmp_line(length('<INVCOVAR>')+1:end),'%d');
    for i=1:N,
      line = line+1;
      tmp_line_next = line_array{line};
      model_struct.state(state_counter).invcovar(mix_counter,i,:) = [ zeros(i-1,1); sscanf(tmp_line_next,'%f')   ]';
    end
  end
  
  gconst_pos = regexpi(tmp_line,'<GCONST>');
  if ~isempty(gconst_pos),
    model_struct.state(state_counter).gconst(mix_counter,:) = sscanf(tmp_line(length('<GCONST>')+1:end),'%f');
  end
  
  transp_pos = regexpi(tmp_line,'<TRANSP>');
  if ~isempty(transp_pos),
    model_struct.A_size = sscanf(tmp_line(length('<TRANSP>')+1:end),'%d');
    for i = 1:model_struct.A_size,
      line = line+1;
      tmp_line_next = line_array{line};
      tmp_vector = sscanf(tmp_line_next,'%f');
      model_struct.A(i,:) = tmp_vector;
    end
    
  end
    
end % for line ...

for i=1:length(model_struct.state),
  if (~(isfield(model_struct.state(i),'mix_weight'))),
    model_struct.state(i).mix_weight(1) = 1;
  end
end

for state_counter=1:length(model_struct.state),	
  for m_counter=1:model_struct.state(state_counter).mix_number
    if ( size(model_struct.state(state_counter).mix_weight) == 0 ) & ( model_struct.state(state_counter).mix_number == 1 )
      model_struct.state(state_counter).mix_weight = 1;
    end
    if (model_struct.state(state_counter).mix_weight(m_counter) == 0),
      model_struct.state(state_counter).mean_vector(m_counter,:) = zeros(size(model_struct.state(state_counter).mean_vector(1,:)));
      model_struct.state(state_counter).variance_vector(m_counter,:) = zeros(size(model_struct.state(state_counter).variance_vector(1,:)));
      model_struct.state(state_counter).gconst(m_counter,:) = zeros(size(model_struct.state(state_counter).gconst(1,:)));
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
