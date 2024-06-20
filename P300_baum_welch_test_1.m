function [answer,staticResults1,NLPResults1] = P300_baum_welch_test_1(szFile,map,TotalTrials,coeff1,feaSelector1)

mapflag=1;
if(nargin < 2)
    mapflag=0;
end;

if ischar(szFile)
    dir = szFile;
    files = ls(strcat(szFile,'\*.dat'));
    szFile = cell(size(files,1),1);
    for i=1:size(files,1)
        szFile{i}=strtrim(strcat(dir,'\',files(i,:)));
    end;
end;

all_signal = cell(size(szFile,1),1);
all_states = cell(size(szFile,1),1);
all_parameters = cell(size(szFile,1),1);
totalLetters = 0;
for iter = 1:size(szFile,1)
    [ signal, states, parameters ] = load_bcidat(szFile{iter});
    all_signal{iter}=signal;
    all_states{iter}=states;
    all_parameters{iter}=parameters;
    totalLetters = totalLetters + length(parameters.TextToSpell.Value{1});
end;

parameters=all_parameters{1};
rate = parameters.SamplingRate.NumericValue;
%epochLength = parameters.EpochLength.NumericValue;
epochLength = 600;
epochPoints = ceil(epochLength * rate / 1000);
if(nargin<3)
    TotalTrials = parameters.NumberOfSequences.NumericValue;
end;
nr=parameters.NumMatrixColumns.NumericValue;
nc=parameters.NumMatrixRows.NumericValue;

DSFactor = 12;
downsampleFunc =@ DownSampleMatByAvg;
%downsampleFunc =@ DownSampleMatByAPCA;
%downsampleFunc =@ DownSampleMatByPivot;
nAvgTrial = 1;

allData   = [];
allStim  = [];
allLabels = [];

for iter = 1:size(szFile,1)
    sprintf('file number %d',iter)
    signal=double(all_signal{iter});
    states=all_states{iter};
    
    stimulusType = states.StimulusType;
    stimulusCode = double(states.StimulusCode);
    stimulusCode(stimulusCode==circshift(stimulusCode,1))=0;
    
    onsetIndices = find(stimulusCode>0);
    allLabels = [allLabels;double(stimulusType(onsetIndices))];
    allStim  = [allStim;stimulusCode(onsetIndices)];
    tempData2 = [];
    
    for chan = 1:size(signal,2)
        tempData = [];
        for i=1:length(onsetIndices)
            tempData=[tempData, DownSample4Feature(signal(onsetIndices(i):(onsetIndices(i)+epochPoints-1),chan),DSFactor,nAvgTrial,[],downsampleFunc)];
        end;
        tempData2=[tempData2;tempData];
    end;
    allData=[allData,tempData2];
end;

word=lower(parameters.TextToSpell.Value{1});
nStim=size(allData,2)/TotalTrials/length(word)/length(szFile);
targets = cell2mat(parameters.TargetDefinitions.Value(:,1));
TotalSet = 1:length(szFile);
staticResults1 = zeros(length(TotalSet)*length(word),length(targets),TotalTrials);
NLPResults1    = zeros(length(TotalSet)*length(word),length(targets),TotalTrials*nStim+1);
answer         = zeros(length(TotalSet)*length(word),1);
EMlength=20;
for i=TotalSet
    sprintf('Training file %d',i)
    fileLength=size(allData,2)/length(TotalSet);
    testIndices=(i-1)*fileLength+(1:fileLength);
    trainData=allData(:,setdiff(1:size(allData,2),testIndices));
    trainLabels=allLabels(setdiff(1:size(allData,2),testIndices));
    
    %[coeff1, feaSelector1] = BuildStepwiseLDA([attr nonr]',[ones(length(attr),1);zeros(length(nonr),1)]);
    attScore1=min(1,max(-1,trainData(feaSelector1,find(trainLabels==1))'*coeff1));
    nonScore1=min(1,max(-1,trainData(feaSelector1,find(trainLabels~=1))'*coeff1));
    attMean1=mean(attScore1);
    nonMean1=mean(nonScore1);
    attSD1=std(attScore1);
    nonSD1=std(nonScore1);
    
    
%     [coeff1,feaSelector1,labels,attMean,nonMean,attSD,nonSD]=EM(trainData,trainLabels,EMlength,.02);
%     attMean1=attMean{EMlength+1};
%     nonMean1=nonMean{EMlength+1};
%     attSD1=attSD{EMlength+1};
%     nonSD1=nonSD{EMlength+1};
%     attP=sum(labels{1}.*labels{EMlength+1})/sum(labels{1});
%     nonP=sum((1-labels{1}).*labels{EMlength+1})/sum(1-labels{1});
    
    TestSet = i;
    sprintf('Testing file %d',i)
    for j=TestSet
        parameters=all_parameters{j};
        word=lower(parameters.TextToSpell.Value{1});
        targets = lower(cell2mat(parameters.TargetDefinitions.Value(:,1)));
        
        p1='_';
        p2='_';
        for k=1:length(word)
            answer((i-1)*length(word)+k)=find(targets==word(k));
            for n=1:length(targets)
                if(mapflag==1)
                    try
                        NLPResults1 ((i-1)*length(word)+k,n,1)=log10(map.(strcat('t',p1,p2,lower(targets(n)))));
                    catch e
                        NLPResults1 ((i-1)*length(word)+k,n,1)=log10(0);
                    end;
                end;
            end;
            NLPResults1((i-1)*length(word)+k,:,1)=NLPResults1((i-1)*length(word)+k,:,1)-log10(sum(power(10,NLPResults1((i-1)*length(word)+k,:,1))));
            for l=1:TotalTrials
                staticResults1((i-1)*length(word)+k,:,l)=staticResults1((i-1)*length(word)+k,:,max(1,l-1));
                for m=1:nStim
                    NLPResults1 ((i-1)*length(word)+k,:,(l-1)*nStim+m+1)=NLPResults1 ((i-1)*length(word)+k,:,(l-1)*nStim+m);
                    index=m+((l-1)+((k-1)+(j-1)*length(word))*TotalTrials)*nStim;
                    score1=min(attMean1+2*attSD1,max(nonMean1-2*nonSD1,allData(feaSelector1,index)'*coeff1));
                    for n=1:length(targets)
                        row=floor((n-1)/nc)+1;
                        col=mod(n-1,nc)+1+nr;
                        if or(allStim(index)==row,allStim(index)==col)
                            staticResults1((i-1)*length(word)+k,n,l)=staticResults1((i-1)*length(word)+k,n,l)+score1;
                            NLPResults1((i-1)*length(word)+k,n,(l-1)*nStim+m+1)=NLPResults1((i-1)*length(word)+k,n,(l-1)*nStim+m+1)+...
                                log10(normpdf(score1,attMean1,attSD1))-log10(normpdf(score1,nonMean1,nonSD1));
                        end;
                    end;
                    NLPResults1((i-1)*length(word)+k,:,(l-1)*nStim+m+1)=NLPResults1((i-1)*length(word)+k,:,(l-1)*nStim+m+1)-log10(sum(power(10,NLPResults1((i-1)*length(word)+k,:,(l-1)*nStim+m+1))));
                end;
            end;
            p1=p2;
            p2=word(k);
        end;
    end;
end;
NLPResults1=power(10,NLPResults1);
for i=1:size(staticResults1,1)
    for j=2:size(staticResults1,3)
        m1=max(staticResults1(i,:,j));
        staticResults1(i,:,j)=staticResults1(i,:,j)*(j-1)/m1;
    end;
end;
