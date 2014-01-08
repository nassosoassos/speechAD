function models2mmf(model_cell, mmfName)
% Function to write a list of models to the mmfName mmf file
mpath = fileparts(mmfName);
[s, m, m_id]=mkdir(mpath);

mmfId = fopen(mmfName, 'w');

n_models = length(model_cell);

for model_counter = 1:n_models 
    current_model = model_cell{model_counter};
    
    current_htk_text = model2htk(current_model);
    fprintf(mmfId, '%s', current_htk_text);
end
fclose(mmfId);

    

function outString = model2htk(model_struct)
% Generate the corresponding htk string for the given model
outString='';

outString = sprintf('%s~o\n',outString);
nStreams = length(model_struct.streams);
nStates = length(model_struct.prior);

outString=sprintf( '%s<STREAMINFO> %d', outString, nStreams);
dims=zeros(1,nStreams);
for s=1:nStreams
    dims(s) = size(model_struct.streams(s).mu, 1);
    outString=sprintf( '%s %d', outString,dims(s));
end
outString=sprintf( '%s\n', outString);

vector_type = model_struct.vector_type;
outString=sprintf( '%s<VECSIZE> %d<NULLD><%s><DIAGC>\n',outString, sum(dims), vector_type);

outString=sprintf( '%s~h "%s"\n', outString, model_struct.name);

outString=sprintf( '%s<BEGINHMM>\n', outString);
outString=sprintf( '%s<NUMSTATES> %d\n', outString, nStates+2);

for k=1:nStates
    outString=sprintf( '%s<STATE> %d\n', outString, k+1);
    outString=sprintf( '%s<SWEIGHTS> %d ', outString, nStreams);
    for s=1:nStreams
        outString=sprintf( '%s%d ', outString, model_struct.streams(s).sweights(k));
    end
    outString=sprintf( '%s\n', outString);
    for s=1:nStreams
        outString=sprintf( '%s<STREAM> %d\n', outString, s);
        meanString=sprintf( '<MEAN> %d\n', dims(s));
        varString=sprintf( '<VARIANCE> %d\n', dims(s));
        for l=1:dims(s)
            meanString=sprintf( '%s%e ', meanString, model_struct.streams(s).mu(l,k));
            varString=sprintf( '%s%e ', varString, model_struct.streams(s).Sigma(l,k));
        end
        meanString=sprintf( '%s\n', meanString);
        varString=sprintf( '%s\n', varString);
        outString=sprintf('%s%s%s', outString, meanString, varString);
    end
end
outString=sprintf('%s<TRANSP> %d\n', outString, nStates+2);

% HTK Transition matrix preparation. Take the null states at the beginning
% and at the end into consideration
transmat = zeros(nStates+2, nStates+2);
transmat(1,2:nStates+1) = model_struct.prior;
transmat(2:nStates+1,nStates+2) = model_struct.exiting_probs(:);
transmat(2:nStates+1, 2:nStates+1) = model_struct.transmat;

for k=1:nStates+2
    for l=1:nStates+2
        outString=sprintf( '%s%e ', outString, transmat(k,l));
    end
    outString=sprintf('%s\n', outString);
end 

outString=sprintf( '%s<ENDHMM>', outString);

