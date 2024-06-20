function [answers, ind_results, pop_results, pop_results2] = P300_test(trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1, map)

mapflag=1;
if(nargin < 7)
    mapflag=0;
end;

ws=zeros(length(trainData1),size(trainData1{1},2));
for i=1:size(ws,1)
    ws(i,feaSelector1{i})=coeff1{i};
end;

a_means=cell(length(trainData1),1);
n_means=cell(length(trainData1),1);
a_covs =cell(length(trainData1),1);
n_covs =cell(length(trainData1),1);
for i=1:length(trainData1)
    scores=max(-1,min(2,trainData1{i}*ws'));
    a_means{i}=mean(scores(labels1{i}==1,:),1);
    n_means{i}=mean(scores(labels1{i}~=1,:),1);
    a_covs{i} =cov( scores(labels1{i}==1,:));
    n_covs{i} =cov( scores(labels1{i}~=1,:));
end;

ind_results =cell(length(trainData1),1);
pop_results =cell(length(trainData1),1);
pop_results2=cell(length(trainData1),1);
answers    =cell(length(trainData1),1);
for z=1:length(trainData1)
    sprintf('testing subject: %d',z)
    all_data=[];
    all_stim=[];
    all_labs=[];
    for i=setdiff(1:length(trainData1),z)
        all_data=[all_data;trainData1{i}];
        all_stim=[all_stim;allStim1{i}];
        all_labs=[all_labs;labels1{i}];
    end;
    ws2=ws(setdiff(1:length(trainData1),z),:);
    scores=max(-1,min(2,all_data*ws2'));
    mean_a_pop=mean(scores(all_labs==1,:),1);
    mean_n_pop=mean(scores(all_labs~=1,:),1);
    cov_a_pop =cov (scores(all_labs==1,:),1);
    cov_n_pop =cov (scores(all_labs~=1,:),1);
    
    start_ind=0;
    NLPResults1 = [];
    NLPResults2 = [];
    NLPResults3 = [];
    answer      = [];
    for i=1:size(parameters1,2)
        if(isempty(parameters1{z,i}))
            break;
        end;
        sprintf('testing file %d for subject %d',i,z)
        parameters=parameters1{z,i};
        word=lower(parameters.TextToSpell.Value{1});
        targets = lower(cell2mat(parameters.TargetDefinitions.Value(:,1)));
        nTargets=length(targets);
        nLetters=length(word);
        nSeq=parameters.NumberOfSequences.NumericValue;
        nr=parameters.NumMatrixRows.NumericValue;
        nc=parameters.NumMatrixColumns.NumericValue;
        nStim=nr+nc;
        nTrials=nLetters*nSeq*nStim;
        
        test_data=trainData1{z}(start_ind+(1:nTrials),:);
        test_stim=allStim1{z}  (start_ind+(1:nTrials),:);
        train_data=trainData1{z}([1:start_ind (start_ind+nTrials+1):end],:);
        train_labs=labels1{z}   ([1:start_ind (start_ind+nTrials+1):end],:);
        start_ind=start_ind+nTrials;
        
        [coeff,  feaSelector] = BuildStepwiseLDA(train_data, train_labs);
        ws1=zeros(1,size(ws2,2));
        ws1(feaSelector)=coeff;
        train_scores=max(-1,min(2,train_data*ws1'));
        mean_a_ind=mean(train_scores(train_labs==1,:),1);
        mean_n_ind=mean(train_scores(train_labs~=1,:),1);
        std_a_ind =std (train_scores(train_labs==1,:),1);
        std_n_ind =std (train_scores(train_labs~=1,:),1);
        
        train_scores2=max(-1,min(2,train_data*ws2'));
        mean_a_pop2=mean(train_scores2(train_labs==1,:),1);
        mean_n_pop2=mean(train_scores2(train_labs~=1,:),1);
        cov_a_pop2 =cov (train_scores2(train_labs==1,:),1);
        cov_n_pop2 =cov (train_scores2(train_labs~=1,:),1);
        
        p1='_';
        p2='_';
        for k=1:nLetters
            answer((i-1)*nLetters+k)=find(targets==word(k));
            for n=1:nTargets
                if(mapflag==1)
                    try
                        NLPResults1 ((i-1)*nLetters+k,n,1)=log10(map.(strcat('t',p1,p2,lower(targets(n)))));
                        NLPResults2 ((i-1)*nLetters+k,n,1)=log10(map.(strcat('t',p1,p2,lower(targets(n)))));
                        NLPResults3 ((i-1)*nLetters+k,n,1)=log10(map.(strcat('t',p1,p2,lower(targets(n)))));
                    catch e
                        NLPResults1 ((i-1)*nLetters+k,n,1)=log10(0);
                        NLPResults2 ((i-1)*nLetters+k,n,1)=log10(0);
                        NLPResults3 ((i-1)*nLetters+k,n,1)=log10(0);
                    end;
                end;
            end;
            NLPResults1((i-1)*nLetters+k,:,1)=NLPResults1((i-1)*nLetters+k,:,1)-log10(sum(power(10,NLPResults1((i-1)*nLetters+k,:,1))));
            NLPResults2((i-1)*nLetters+k,:,1)=NLPResults2((i-1)*nLetters+k,:,1)-log10(sum(power(10,NLPResults2((i-1)*nLetters+k,:,1))));
            NLPResults3((i-1)*nLetters+k,:,1)=NLPResults3((i-1)*nLetters+k,:,1)-log10(sum(power(10,NLPResults3((i-1)*nLetters+k,:,1))));
            for l=1:nSeq
                for m=1:nStim
                    if(and(z==2,and(((i-1)*nLetters+k)==37,((l-1)*nStim+m+1)==176)))
                        
                    end;
                    NLPResults1 ((i-1)*nLetters+k,:,(l-1)*nStim+m+1)=NLPResults1 ((i-1)*nLetters+k,:,(l-1)*nStim+m);
                    NLPResults2 ((i-1)*nLetters+k,:,(l-1)*nStim+m+1)=NLPResults2 ((i-1)*nLetters+k,:,(l-1)*nStim+m);
                    NLPResults3 ((i-1)*nLetters+k,:,(l-1)*nStim+m+1)=NLPResults3 ((i-1)*nLetters+k,:,(l-1)*nStim+m);
                    index=m+((l-1)+(k-1)*nSeq)*nStim;
                    score1=max(-1,min(2,test_data(index,:)*ws1'));
                    score2=max(-1,min(2,test_data(index,:)*ws2'));
                    stim=test_stim(index);
                    for n=1:nTargets
                        row=floor((n-1)/nc)+1;
                        col=mod(n-1,nc)+1+nr;
                        if or(stim==row,stim==col)
                            NLPResults1((i-1)*nLetters+k,n,(l-1)*nStim+m+1)=NLPResults1((i-1)*nLetters+k,n,(l-1)*nStim+m+1)+...
                                log10(normpdf(score1,mean_a_ind,std_a_ind))-log10(normpdf(score1,mean_n_ind,std_n_ind));
                            NLPResults2((i-1)*nLetters+k,n,(l-1)*nStim+m+1)=NLPResults2((i-1)*nLetters+k,n,(l-1)*nStim+m+1)+...
                                log10(mvnpdf (score2,mean_a_pop,cov_a_pop))-log10(mvnpdf (score2,mean_n_pop,cov_n_pop));
                            NLPResults3((i-1)*nLetters+k,n,(l-1)*nStim+m+1)=NLPResults3((i-1)*nLetters+k,n,(l-1)*nStim+m+1)+...
                                log10(mvnpdf(score2,mean_a_pop2,cov_a_pop2))-log10(mvnpdf(score2,mean_n_pop2,cov_n_pop2));
                        end;
                    end;
                    NLPResults1((i-1)*nLetters+k,:,(l-1)*nStim+m+1)=NLPResults1((i-1)*nLetters+k,:,(l-1)*nStim+m+1)-...
                        log10(sum(power(10,NLPResults1((i-1)*nLetters+k,:,(l-1)*nStim+m+1))));
                    NLPResults2((i-1)*nLetters+k,:,(l-1)*nStim+m+1)=NLPResults2((i-1)*nLetters+k,:,(l-1)*nStim+m+1)-...
                        log10(sum(power(10,NLPResults2((i-1)*nLetters+k,:,(l-1)*nStim+m+1))));
                    NLPResults3((i-1)*nLetters+k,:,(l-1)*nStim+m+1)=NLPResults3((i-1)*nLetters+k,:,(l-1)*nStim+m+1)-...
                        log10(sum(power(10,NLPResults3((i-1)*nLetters+k,:,(l-1)*nStim+m+1))));
                end;
            end;
            p1=p2;
            p2=word(k);
        end;
    end;
    ind_results{z} =power(10,NLPResults1);
    pop_results{z} =power(10,NLPResults2);
    pop_results2{z}=power(10,NLPResults3);
    answers{z}    =answer;
end;