function kind_code = htk_kind_code(kind)
% HTK_KIND_CODE - Return the code of the given HTK data kind
% The code is the sum of the following values:   
%			0		WAVEFORM
%			1		LPC
%			2		LPREFC
%			3		LPCEPSTRA
%			4		LPDELCEP
%			5		IREFC
%			6		MFCC
%			7		FBANK
%			8		MELSPEC
%			9		USER
%			10		DISCRETE
%			64		-E		Includes energy terms
%			128	_N		Suppress absolute energy
%			256	_D		Include delta coefs
%			512	_A		Include acceleration coefs
%			1024	_C		Compressed (not implemented yet)
%			2048	_Z		Zero mean static coefs
%			4096	_K		CRC checksum (not implemented yet)
%			8192	_0		Include 0'th cepstral coef
htk_kinds.WAVEFORM = 0;
htk_kinds.LPC = 1;
htk_kinds.LPREFC = 2;
htk_kinds.LPCEPSTRA = 3;
htk_kinds.LPDELCEP = 4;
htk_kinds.IREFC = 5;
htk_kinds.MFCC = 6;
htk_kinds.FBANK = 7;
htk_kinds.MELSPEC = 8;
htk_kinds.USER = 9;
htk_kinds.LSP = 9;
htk_kinds.DISCRETE = 10;

htk_props = {'E','N','D','A','C','Z','K','0'};

kind_comps = strread(kind,'%s','delimiter','_');

if (isfield(htk_kinds,kind_comps{1})) 
  kind_code = htk_kinds.(kind_comps{1});
  for k=2:length(kind_comps) 
    pos = strmatch(kind_comps{k},htk_props);
    kind_code = kind_code + 2^(5+pos);
  end
else
  error('Unknown Format.');
end
