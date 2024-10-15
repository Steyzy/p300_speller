function [output,nStims]=pf_classify(signal,stimulusCode,stimulusType,phaseInSequence,word,rate,epochLength,thresh,nr,nc,targets,corpus_dir,nPart)

[num,den]=butter(1,1/(rate/2),'high'); %high pass the signal
signal=filter(num,den,double(signal));

epochPoints = ceil(epochLength * rate / 1000); %the number of points used in classification

DSFactor = 12; %parameters for down sampling
downsampleFunc =@ DownSampleMatByAvg;
nAvgTrial = 1;


stimulusCode(stimulusCode==circshift(stimulusCode,1))=0; %finding onset indices and labels for evoked responses
onsetIndices = find(stimulusCode>0);

ends=unique([1;find(and(phaseInSequence==3,circshift(phaseInSequence,1)~=3))]);
onsetIndices=onsetIndices(onsetIndices<ends(end));
tnStim=zeros(length(ends)-1,1);
for i=2:length(ends)
    tnStim(i-1)=sum(and(onsetIndices>ends(i-1),onsetIndices<ends(i)));
end;
onsetIndices=onsetIndices(1:sum(tnStim));
labels = double(stimulusType(onsetIndices));
stim  = stimulusCode(onsetIndices);

nLetters = length(word); %number of letters to classify
rows=floor(((1:length(targets))-1)/nc)+1; %row labels for each letter
cols=mod((1:length(targets))-1,nc)+1+nr; %column labels for each letter

data=zeros(floor(epochPoints/DSFactor)*size(signal,2),length(onsetIndices)); %get responses for each stimulus
for i=1:length(onsetIndices)
    temp=DownSample4Feature(signal(onsetIndices(i):(onsetIndices(i)+epochPoints-1),:),DSFactor,nAvgTrial,[],downsampleFunc);
    data(:,i)=temp(:);
end;


answer=zeros(nLetters,1);
nStims=zeros(nLetters,1);

pfModel=pf_model.pf_model(nPart,corpus_dir,targets); %create the particle filter

index=1;
for i=1:nLetters %cross validate by letter
    fprintf('Training fold %d\n',i)
    letterLength=size(data,2)/nLetters;
    testIndices=(i-1)*letterLength+(1:letterLength); %indices for the test letter
    trainData=data(:,setdiff(1:size(data,2),testIndices)); %data for the training letters 
    trainLabels=labels(setdiff(1:size(data,2),testIndices)); %labels for the training letters
    
    [coeff1,feaSelector1]=BuildStepwiseLDA(trainData',trainLabels); %train LDA on the training letters
    attScore1=min(1,max(-1,trainData(feaSelector1,find(trainLabels==1))'*coeff1)); %convert the training signals into scores
    nonScore1=min(1,max(-1,trainData(feaSelector1,find(trainLabels~=1))'*coeff1));
    
    attMean=mean(attScore1); %mean of the attended scores
    nonMean=mean(nonScore1); %mean of the non-attended scores
    attSD=std(attScore1); %standard deviation of the attended scores
    nonSD=std(nonScore1); %standard deviation of the non-attended scores
    
    fprintf('Testing fold %d\n',i)
    
    
    answer(i)=targets(targets==word(i));
    fprintf(['Target character: ' char(answer(i)) '\n'])
    
    pfModel.project;
    nStims(i)=0;
    dist=pfModel.get_dist;
    while((max(dist)<thresh)&&(nStims(i)<tnStim(i))) %add stimulus responses until the decision threshold is exceeded
        nStims(i)=nStims(i)+1;
        fprintf('    Stimulus number %d\n',nStims(i));
        ind=index+nStims(i);
        
        score=min(1,max(-1,data(feaSelector1,ind)'*coeff1)); %convert response into score
        prob=normpdf(score,attMean,attSD)/normpdf(score,nonMean,nonSD); %convert score into probability
        probs=ones(length(targets),1); %create probability distribution across letters
        probs(stim(ind)==rows)=prob; 
        probs(stim(ind)==cols)=prob;
        probs=probs/sum(probs);
        pfModel.update_dist(probs); %update the weights of the model based on the probability distribution
        dist=pfModel.get_dist; %get the distribution across letters
    end;
    index=index+tnStim(i);
    
    pfModel.resample; %resample the particles
    output=pfModel.get_results; %get results to this point
    [output.keys' output.values'] %output results to screen
end;
