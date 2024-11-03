function [answers, all_results, all_accs, all_coeffs, all_feaSelectors, all_mean_as, all_mean_ns, all_std_as, all_std_ns, all_trells, all_labels] = P300_baum_welch(z, nLetters_range, nSeq_range, trainData1, allStim1, parameters1, labels1, exclude, ws, parameters, cv, blank, transition_matrix)

maxLetters=length(nLetters_range);
maxSeqs=length(nSeq_range);
answers=cell(maxLetters,maxSeqs);
all_coeffs=cell(maxLetters,maxSeqs);
all_feaSelectors=cell(maxLetters,maxSeqs);
all_mean_as=cell(maxLetters,maxSeqs);
all_mean_ns=cell(maxLetters,maxSeqs);
all_std_as=cell(maxLetters,maxSeqs);
all_std_ns=cell(maxLetters,maxSeqs);
all_results=cell(maxLetters,maxSeqs);
all_accs=cell(maxLetters,maxSeqs);
all_trells=cell(maxLetters,maxSeqs);
all_labels=cell(maxLetters,maxSeqs);

min_score=-1;
max_score=2;
max_iter=10;    %manually set max iteration to 10

sprintf('testing subject: %d',z)
len = length(cv)-1;
range = setdiff(cv, z);
info_array = zeros(len, 2);
sum_dim = 0;
for k=1:len
    j = range(k);
    info_array(k, 2) = size(trainData1{j}, 1);  %dim1 of trainData1{j}
    if k>1
        info_array(k, 1) = info_array(k-1, 1)+info_array(k-1, 2);
    else
        info_array(k, 1) = 1;
    end
    sum_dim = sum_dim+info_array(k, 2);
end

tic;
dim2 = size(trainData1{1}, 2);  %dim2 of trainData1{i}
all_data = zeros(sum_dim, dim2);
all_stim = zeros(sum_dim, 1);
all_labs = zeros(sum_dim, 1);
for i=1:len
    j = range(i);
    start = info_array(i, 1);
    duration = info_array(i, 2);
    all_data(start:start+duration-1, :) = trainData1{j};
    all_stim(start:start+duration-1, :) = allStim1{j};
    all_labs(start:start+duration-1, :) = labels1{j};
end
elapsedTime = toc; % Stop timer and get elapsed time
fprintf('data loading time is %.4f seconds.\n', elapsedTime);

test_data=trainData1{z};
w0=ws(setdiff(cv,z),:);
scores_pop=max(min_score,min(max_score,all_data*w0'));
mean_a_pop=mean(scores_pop(all_labs==1,:),1);
mean_n_pop=mean(scores_pop(all_labs~=1,:),1);
cov_a_pop =cov (scores_pop(all_labs==1,:),1);
cov_n_pop =cov (scores_pop(all_labs~=1,:),1);

%set up parameters to properly extract a subset of data
fSeq=parameters.NumberOfSequences.NumericValue;     %stands for full seq
nr=parameters.NumMatrixRows.NumericValue;
nc=parameters.NumMatrixColumns.NumericValue;
nStim=nr+nc;

for nLetters=nLetters_range
    for nSeq=nSeq_range

        %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        % taking slices of test_data by changing nLetters
        %!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
        %take slices of test_data and allStim1{z} for unsupervised training
        test_data_sliced = [];
        z_stim_sliced = [];
        start=1;
        for i=1:size(parameters1,2)
            params=parameters1{z,i};            
            if(isempty(params))
                break;
            end
            word=lower(params.TextToSpell.Value{1});
            temp=start;
            for j=1:nLetters
                test_data_sliced = [test_data_sliced; test_data(temp:(temp+nStim*nSeq-1),:)];
                z_stim_sliced = [z_stim_sliced; allStim1{z}(temp:(temp+nStim*nSeq-1),:)];
                temp=temp+nStim*fSeq;
            end
            start=start+nStim*fSeq*length(word);
        end    
    
        %Gaussian mixture model based on mean and cov
        scores=max(-1,min(2,test_data_sliced*w0'));
        try
            probs_a=mvnpdf(scores,mean_a_pop,cov_a_pop);
            probs_n=mvnpdf(scores,mean_n_pop,cov_n_pop);
        catch
            scores=normrnd(zeros(length(scores),1),1);
            probs_a=normpdf(scores,0,1);
            probs_n=normpdf(scores,0,1);
        end
    
        % set min non-zero value to 0
        mv=min([probs_a(probs_a>0);probs_n(probs_n>0)]);
        probs_a(probs_a==0)=mv;
        probs_n(probs_n==0)=mv;
    
    
        %=======================================================================
        %baum-welch algorith, updating the parameters
        %=======================================================================
        converged=0;
        counter=1;
        coeffs=cell(0);
        feaSelectors=cell(0);
        mean_as=cell(0);
        mean_ns=cell(0);
        std_as=cell(0);
        std_ns=cell(0);
        accs=[];
        trells=cell(0);
        labs=cell(0);
        answer=[];
        used=zeros(size(scores,1),1);
    
        tic;
        while ~converged
            sprintf('iteration: %d',counter)
            fb_trellis=[];
            results=[];
            tLetters=0;
            for i=1:size(parameters1,2)
                if(isempty(parameters1{z,i}))
                    break;
                end
                parameters=parameters1{z,i};
                word=lower(parameters.TextToSpell.Value{1});
                parameters.TargetDefinitions.Value{36,1}=' ';
                targets = lower(cell2mat(parameters.TargetDefinitions.Value(:,1)));
                nTargets=length(targets);
                tLetters=tLetters+nLetters;
                % nSeq=parameters.NumberOfSequences.NumericValue;
                nr=parameters.NumMatrixRows.NumericValue;
                nc=parameters.NumMatrixColumns.NumericValue;
                nStim=nr+nc;
    
                a_trellis=zeros(nLetters,power(nTargets,2));
                b_trellis=zeros(nLetters,power(nTargets,2));
                v_trellis=zeros(nLetters,power(nTargets,2));
                back_pointers=zeros(nLetters,power(nTargets,2));
                target=mod((1:size(a_trellis,2))-1,nTargets)+1;
    
                for j=1:nLetters
                    answer(tLetters-nLetters+j)=find(targets==word(j));
                    a_score=zeros(1,size(a_trellis,2));
                    b_score=zeros(1,size(b_trellis,2));
                    for k=1:nStim*nSeq
                        index=(tLetters-nLetters+j-1)*nStim*nSeq+k;
                        if(isempty(find(exclude==i,1)))
                            used(index)=1;
                        end
                        attended=or(floor((target-1)/nr+1)==z_stim_sliced(index),...
                            (mod((target-1),nr)+nr+1)==z_stim_sliced(index));
                        prob=attended*(probs_a(index)/probs_n(index))+(1-attended);
                        a_score=a_score+log10(prob);
                        a_score=a_score-max(a_score);
    
                        if j>1
                            index2=(tLetters+1-j)*nStim*nSeq+k;
                            attended2=or(floor((target-1)/nr+1)==z_stim_sliced(index2),...
                                (mod((target-1),nr)+nr+1)==z_stim_sliced(index2));
                            prob2=attended2*(probs_a(index2)/probs_n(index2))+(1-attended2);
                            b_score=b_score+log10(prob2);
                            b_score=b_score-max(b_score);
                        end
                    end
                    if j>1
                        a_trellis(j,:)=log10(power(10,a_trellis(j-1,:))*transition_matrix)+a_score;
                        %a_trellis(j,:)=a_trellis(j,:)-max(a_trellis(j,:));
                        b_trellis(nLetters+1-j,:)=log10(power(10,b_trellis(nLetters+2-j,:)+b_score)*transition_matrix');
                        [m_score,back_pointer]=max(v_trellis(j-1,:)'*ones(1,length(blank))+transition_matrix,[],1);
                        back_pointers(j,:)=back_pointer;
                        v_trellis(j,:)=a_score+m_score;
                        %v_trellis(j,:)=v_trellis(j,:)-max(v_trellis(j,:));
                    else
                        a_trellis(j,:)=log10(blank*transition_matrix)+a_score;
                        b_trellis(nLetters+1-j,:)=0;
                        [m_score,~]=max(log10(blank')*ones(1,length(blank))+transition_matrix,[],1);
                        v_trellis(j,:)=a_score+m_score;
                    end
                end
                [~,b]=max(v_trellis(nLetters,:));   % get the last char
                result=b;
                for j=nLetters:-1:2
                    b=back_pointers(j,b);   % trace back the best path
                    result=[b,result];
                end
                results=[results,mod(result-1,nTargets)+1];
                fb_trellis=[fb_trellis; a_trellis+b_trellis];
            end
            fb_trellis=fb_trellis-max(fb_trellis(:));
            fb_trellis=power(10,fb_trellis);
            fb_trellis=fb_trellis./(sum(fb_trellis,2)*ones(1,size(fb_trellis,2)));
            lab_pdf=sum(reshape(fb_trellis,[size(fb_trellis,1),nTargets,nTargets]),3);
            lab_pdf=lab_pdf(floor(((1:size(scores,1))-1)/nSeq/nStim+1),:);
            lab_cdf=cumsum(lab_pdf,2);
    
            r=rand(size(lab_cdf,1),1)*ones(1,nTargets);
            sample=sum(lab_cdf<r,2)+1;
            lab=or(floor((sample-1)/nr+1)==z_stim_sliced,(mod((sample-1),nr)+nr+1)==z_stim_sliced);
    
            [coeff, feaSelector] = BuildStepwiseLDA(test_data_sliced(find(used==1),:), lab(find(used==1)));
            scores=test_data_sliced(:,feaSelector)*coeff;
            mean_a=mean(scores((used.*lab)    ==1,:),1);
            mean_n=mean(scores((used.*(1-lab))==1,:),1);
            std_a =std (scores((used.*lab)    ==1,:),1);
            std_n =std (scores((used.*(1-lab))==1,:),1);
            probs_a=normpdf(scores,mean_a,std_a);
            probs_n=normpdf(scores,mean_n,std_n);
    
            acc=sum(answer==results)/length(answer);
    
            coeffs{counter}=coeff;
            feaSelectors{counter}=feaSelector;
            mean_as{counter}=mean_a;
            mean_ns{counter}=mean_n;
            std_as{counter}=std_a;
            std_ns{counter}=std_n;
            accs=[accs, acc];
            accs
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
            end
            counter=counter+1;
        end
    
        elapsedTime = toc; % Stop timer and get elapsed time
        fprintf('baum-welch time for %d letters and %d sequences is %.4f seconds.\n', nLetters, nSeq, elapsedTime);
    
        all_coeffs{nLetters,nSeq}=coeffs;
        all_feaSelectors{nLetters,nSeq}=feaSelectors;
        all_mean_as{nLetters,nSeq}=mean_as;
        all_mean_ns{nLetters,nSeq}=mean_ns;
        all_std_as{nLetters,nSeq}=std_as;
        all_std_ns{nLetters,nSeq}=std_ns;
        all_accs{nLetters,nSeq}=accs;
        answers{nLetters,nSeq}=answer;
        all_results{nLetters,nSeq}=results;
        all_trells{nLetters,nSeq}=trells;
        all_labels{nLetters,nSeq}=labs;
    end
end

