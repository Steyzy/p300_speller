function [answers, all_results, all_accs, all_coeffs, all_feaSelectors, all_mean_as, all_mean_ns, all_std_as, all_std_ns, all_channels, all_trells, all_labels] = P300_baum_welch(trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1, map, subject_index, exclude)

min_score=-1;
max_score=2;
max_iter=3;

ws=zeros(length(trainData1),size(trainData1{1},2));
for i=1:size(ws,1)
    ws(i,feaSelector1{i})=coeff1{i};
end;

sprintf('compiling transition matrix')
parameters=parameters1{1,1};
targets = lower(cell2mat(parameters.TargetDefinitions.Value(:,1)));
transition_matrix=zeros(length(targets)*length(targets));
lambda=1; % add lambda smoothing
for i=1:length(targets)
    for j=1:length(targets)
        for k=1:length(targets)
            denom=length(targets)*lambda;
            try
                denom=map.(['t' targets([i j])' 'X'])+length(targets)*lambda;
            catch e
            end;
            try
                denom=denom-map.(['t' targets([i j])' '0']);
            catch e
            end;
            num=lambda;
            try
                num=map.(['t' targets([i j k])'])+lambda;
            catch e
            end;
            transition_matrix((i-1)*length(targets)+j,(j-1)*length(targets)+k)=num/denom;
        end;
    end;
end;
blank=zeros(1,length(targets)*length(targets));
blank(end)=1;

answers =cell(length(trainData1),1);
all_coeffs=cell(length(trainData1),1);
all_feaSelectors=cell(length(trainData1),1);
all_mean_as=cell(length(trainData1),1);
all_mean_ns=cell(length(trainData1),1);
all_std_as=cell(length(trainData1),1);
all_std_ns=cell(length(trainData1),1);
all_results=cell(length(trainData1),1);
all_channels=cell(length(trainData1),1);
all_accs=cell(length(trainData1),1);
all_trells=cell(length(trainData1),1);
all_labels=cell(length(trainData1),1);

zs=1:length(trainData1);
if(nargin>7) %for running single subject
    zs=subject_index;
end;
for z=zs
    sprintf('testing subject: %d',z)
    all_data=[];
    all_stim=[];
    all_labs=[];
    for i=setdiff(1:length(trainData1),z)
        all_data=[all_data;trainData1{i}];
        all_stim=[all_stim;allStim1{i}];
        all_labs=[all_labs;labels1{i}];
    end;
    test_data=trainData1{z};
    w0=ws(setdiff(1:length(trainData1),z),:);
    scores_pop=max(min_score,min(max_score,all_data*w0'));
    mean_a_pop=mean(scores_pop(all_labs==1,:),1);
    mean_n_pop=mean(scores_pop(all_labs~=1,:),1);
    cov_a_pop =cov (scores_pop(all_labs==1,:),1);
    cov_n_pop =cov (scores_pop(all_labs~=1,:),1);
    
%     temp=[];
%     temp(:,:)=sum(reshape(w0.^2,[size(w0,1),13,32]),2);
%     temp2=temp./(sum(temp,2)*ones(1,32));
%     [channels,~]=ttest(temp2,[],.05);
%     w0=reshape(w0,[size(w0,1),13,32]);
%     w0=w0(:,:,find(channels==1));
%     w0=reshape(w0,[size(w0,1),size(w0,2)*size(w0,3)]);
%     test_data=reshape(test_data,[size(test_data,1),13,32]);
%     test_data=test_data(:,:,find(channels==1));
%     test_data=reshape(test_data,[size(test_data,1),size(test_data,2)*size(test_data,3)]);
    
    scores=max(-1,min(2,test_data*w0'));
    try
    probs_a=mvnpdf(scores,mean_a_pop,cov_a_pop);
    probs_n=mvnpdf(scores,mean_n_pop,cov_n_pop);
    catch
        scores=normrnd(zeros(length(scores),1),1);
        probs_a=normpdf(scores,0,1);
        probs_n=normpdf(scores,0,1);
    end;
    
%     probs_a=zeros(size(probs_a));
%     probs_n=zeros(size(probs_n));
    
    converged=0;
    counter=1;
    coeffs=cell(0);
    feaSelectors=cell(0);
    mean_as=cell(0);
    mean_ns=cell(0);
    std_as=cell(0);
    std_ns=cell(0);
    trells=cell(0);
    labs=cell(0);
    answer=[];
    used=zeros(size(scores,1),1);
    while ~converged
        sprintf('iteration: %d',counter)
        fb_trellis=[];
        results=[];
        is=1:size(parameters1,2);
        for i=1:size(parameters1,2)
            if(isempty(parameters1{z,i}))
                break;
            end;
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
            
            a_trellis=zeros(nLetters,power(nTargets,2));
            b_trellis=zeros(nLetters,power(nTargets,2));
            v_trellis=zeros(nLetters,power(nTargets,2));
            back_pointers=zeros(nLetters,power(nTargets,2));
            target=mod((1:size(a_trellis,2))-1,nTargets)+1;
            
            for j=1:nLetters
                answer((i-1)*nLetters+j)=find(targets==word(j));
                a_score=zeros(1,size(a_trellis,2));
                b_score=zeros(1,size(b_trellis,2));
                for k=1:nStim*nSeq
                    index=((i-1)*nLetters+j-1)*nStim*nSeq+k;
                    if(isempty(find(exclude==i,1)))
                        used(index)=1;
                    end;
                    attended=or(floor((target-1)/nr+1)==allStim1{z}(index),...
                        (mod((target-1),nr)+nr+1)==allStim1{z}(index));
                    prob=attended*(probs_a(index)/probs_n(index))+(1-attended);
                    a_score=a_score+log10(prob);
                    
                    if j>1
                        index2=(i*nLetters+1-j)*nStim*nSeq+k;
                        attended2=or(floor((target-1)/nr+1)==allStim1{z}(index2),...
                            (mod((target-1),nr)+nr+1)==allStim1{z}(index2));
                        prob2=attended2*(probs_a(index2)/probs_n(index2))+(1-attended2);
                        b_score=b_score+log10(prob2);
                    end;
                end;
                if j>1
                    a_trellis(j,:)=log10(power(10,a_trellis(j-1,:))*transition_matrix)+a_score;
                    b_trellis(nLetters+1-j,:)=log10(power(10,b_trellis(nLetters+2-j,:)+b_score)*transition_matrix');
                    [m_score,back_pointer]=max(v_trellis(j-1,:)'*ones(1,length(blank))+transition_matrix,[],1);
                    back_pointers(j,:)=back_pointer;
                    v_trellis(j,:)=a_score+m_score;
                else
                    a_trellis(j,:)=log10(blank*transition_matrix)+a_score;
                    b_trellis(nLetters+1-j,:)=0;
                    [m_score,~]=max(log10(blank')*ones(1,length(blank))+transition_matrix,[],1);
                    v_trellis(j,:)=a_score+m_score;
                end;
            end;
            [~,b]=max(v_trellis(nLetters,:));
            result=b;
            for j=nLetters:-1:2
                b=back_pointers(j,b);
                result=[b,result];
            end;
            results=[results,mod(result,nTargets)];
            fb_trellis=[fb_trellis; a_trellis+b_trellis];
        end;
        fb_trellis=fb_trellis-max(fb_trellis(:));
        fb_trellis=power(10,fb_trellis);
        fb_trellis=fb_trellis./(sum(fb_trellis,2)*ones(1,size(fb_trellis,2)));
        lab_pdf=sum(reshape(fb_trellis,[size(fb_trellis,1),nTargets,nTargets]),3);
        lab_pdf=lab_pdf(floor(((1:size(scores,1))-1)/nSeq/nStim+1),:);
        lab_cdf=cumsum(lab_pdf,2);
        
        r=rand(size(lab_cdf,1),1)*ones(1,nTargets);
        sample=sum(lab_cdf<r,2)+1;
        lab=or(floor((sample-1)/nr+1)==allStim1{z},(mod((sample-1),nr)+nr+1)==allStim1{z});

        [coeff,  feaSelector] = BuildStepwiseLDA(test_data(find(used==1),:), lab(find(used==1)));
        scores=test_data(:,feaSelector)*coeff;
        mean_a=mean(scores((used.*lab)    ==1,:),1);
        mean_n=mean(scores((used.*(1-lab))==1,:),1);
        std_a =std (scores((used.*lab)    ==1,:),1);
        std_n =std (scores((used.*(1-lab))==1,:),1);
        probs_a=normpdf(scores,mean_a,std_a);
        probs_n=normpdf(scores,mean_n,std_n);
        
        acc=sum(answer==results)/length(answer)
        
        coeffs{counter}=coeff;
        feaSelectors{counter}=feaSelector;
        mean_as{counter}=mean_a;
        mean_ns{counter}=mean_n;
        std_as{counter}=std_a;
        std_ns{counter}=std_n;
        accs{counter}=acc;
        trells{counter}=fb_trellis;
        labs{counter}=lab;
        
        
%         for i=1:counter-1
%             if length(coeff)==length(coeffs{i})
%                 if sum(coeff==coeffs{i})==length(coeff)
%                     converged=1;
%                 end;
%             end;
%         end;
        if(counter>=max_iter)
            converged=1;
        end;
        counter=counter+1;
    end;
    all_coeffs{z}=coeffs;
    all_feaSelectors{z}=feaSelectors;
    all_mean_as{z}=mean_as;
    all_mean_ns{z}=mean_ns;
    all_std_as{z}=std_as;
    all_std_ns{z}=std_ns;
    all_accs{z}=accs;
    answers{z}=answer;
    all_results{z}=results;
    all_trells{z}=trells;
    all_labels{z}=labs;
    %all_channels{z}=channels;
end;