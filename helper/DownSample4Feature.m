function [ asig, t, segment] = DownSample4Feature(sig, DSFactor, nAvgTrial, Pivot, fnDownsample)
%
%
%  Downsample signal by DSFactor using the adaptive piecewise linear
%  approximation by Dr.Keogh

if nargin < 3
    nAvgTrial = 1;
end;

segment = [];


if (~isstruct(sig))
    numSegment = fix(size(sig, 1) / DSFactor);
    nTrials = size(sig, 2); ix = 1:nAvgTrial:nTrials; if( nTrials -  ix(end) ~= nAvgTrial-1)ix = ix(1:end-1);end;
    nSmp = size(sig, 1);
    bsig = ones(nSmp, numel(ix));
    for i=1:numel(ix)
        bsig(:,i) = mean(sig(:, ix(i):ix(i)+nAvgTrial-1),2);
    end;
    [asig, t, segment] = feval(fnDownsample, bsig, numSegment);
else
    VarName = fieldnames(sig);
    aCodeSig = sig.(VarName{1});
    ChanName =  fieldnames(aCodeSig);
    ChanSig = aCodeSig.(ChanName{1});
    nRC = length(VarName);      %number of rows/cols;
    nChan = length(ChanName);
    nTrials = size(ChanSig, 2);
    if(strcmpi(func2str(fnDownsample), 'DownSampleMatByPivot')==1)
        nSmp = length(Pivot{1});
    else
        nSmp = fix(size(ChanSig, 1) / DSFactor);
    end;
    ix = 1:nAvgTrial:nTrials; if( nTrials -  ix(end) ~= nAvgTrial-1)ix = ix(1:end-1);end;
    nOrigSmp = size(ChanSig, 1);
    bsig = ones(nOrigSmp, numel(ix));
    nTrials = size(bsig, 2);
%    asig = ones(nTrials * nRC,  nSmp * nChan);
    for i=1:length(VarName)
        aCodeSig = sig.(VarName{i});
        ChanName =  fieldnames(aCodeSig);
        tsig = [];
        for k=1:length(ChanName)
            ChanSig = aCodeSig.(ChanName{k});
            afea = [];
            numSegment = fix(size(ChanSig, 1) / DSFactor);
            for kk=1:numel(ix)
                bsig(:,kk) = mean(ChanSig(:, ix(kk):ix(kk)+nAvgTrial-1),2);
            end;            
            if( strcmpi(func2str(fnDownsample), 'DownSampleMatByPivot')==1)
                afea = feval(fnDownsample, bsig, Pivot{k}); 
            else
                [afea]  = feval(fnDownsample, bsig, numSegment); 
            end;
            tsig = [tsig afea'];
        end;
        rowIdx = (i-1)*nTrials+1: i*nTrials;
        asig(rowIdx, :) =  tsig;
    end;
    t = 1:DSFactor:length(ChanSig);
end;



function [asig, t, segment] = DownSampleMat(sig, numSegment)

for k=1:size(sig, 2)
    asegment = apca(sig(:, k), numSegment);
    for i=1:numSegment
        asig(i, k) = asegment(i).y;
        t(i, k) = asegment(i).lx + (asegment(i).rx - asegment(i).lx) / 2;
    end;
    if nargout > 2
        segment{k} = asegment;
    end;
end;