function mhmm2mmf(prod_mhmm_struct, mmfName)
% Function to write an HMM to the mmfName mmf file
mpath = fileparts(mmfName);
[s, m, m_id]=mkdir(mpath);

mmfId = fopen(mmfName, 'w');
outString='';

outString = sprintf('%s~o\n',outString);

nStreams = length(prod_mhmm_struct.streams);
nStates = length(prod_mhmm_struct.prior);

outString=sprintf( '%s<STREAMINFO> %d', outString, nStreams);
dims=zeros(1,nStreams);
for s=1:nStreams
    dims(s) = size(prod_mhmm_struct.streams(s).mu, 1);
    outString=sprintf( '%s %d', outString,dims(s));
end
outString=sprintf( '%s\n', outString);

vector_type = prod_mhmm_struct.vector_type;
outString=sprintf( '%s<VECSIZE> %d<NULLD><%s><DIAGC>\n',outString, sum(dims), vector_type);

outString=sprintf( '%s~h "%s"\n', outString, prod_mhmm_struct.name);

outString=sprintf( '%s<BEGINHMM>\n', outString);
outString=sprintf( '%s<NUMSTATES> %d\n', outString, nStates+2);

for k=1:nStates
    outString=sprintf( '%s<STATE> %d\n', outString, k+1);
    outString=sprintf( '%s<SWEIGHTS> %d\n', outString, nStreams);
    for s=1:nStreams
        outString=sprintf( '%s%d ', outString, prod_mhmm_struct.streams(s).sweights(k));
    end
    outString=sprintf( '%s\n', outString);
    for s=1:nStreams
        outString=sprintf( '%s<STREAM> %d\n', outString, s);
        meanString=sprintf( '<MEAN> %d\n', dims(s));
        varString=sprintf( '<VARIANCE> %d\n', dims(s));
        for l=1:dims(s)
            meanString=sprintf( '%s%e ', meanString, prod_mhmm_struct.streams(s).mu(l,k));
            varString=sprintf( '%s%e ', varString, prod_mhmm_struct.streams(s).Sigma(l,k));
        end
        meanString=sprintf( '%s\n', meanString);
        varString=sprintf( '%s\n', varString);
        outString=sprintf('%s%s%s', outString, meanString, varString);
    end
end
outString=sprintf('%s<TRANSP> %d\n', outString, nStates+2);

% HTK Transition matrix preparation. Take the null states at the beginning
% and at the end into consideration
% transmat = zeros(nStates+2, nStates+2);
% transmat(1, 2:nStates+1) = prod_mhmm_struct.prior;
% transmat(2:nStates+1, 2:nStates+1) = prod_mhmm_struct.transmat;
% transmat(nStates+1, nStates+1) = 0.5;
% transmat(nStates+1, nStates+2) = 0.5;
transmat = prod_mhmm_struct.transmat;

for k=1:nStates+2
    for l=1:nStates+2
        outString=sprintf( '%s%e ', outString, transmat(k,l));
    end
    outString=sprintf('%s\n', outString);
end 

outString=sprintf( '%s<ENDHMM>', outString);

fprintf(mmfId, '%s', outString);
fclose(mmfId);
