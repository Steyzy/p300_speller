addpath('helper');
szFile = '/Users/yangziyi/Desktop/Neuro Research/p300_data/';

sprintf('loading stepwise regression data')
try
    load('/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/all_subjects.mat');
catch e
    [all_files, trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1] = P300_train(szFile);
    save('/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/all_subjects.mat','all_files','trainData1','allStim1','parameters1','labels1','coeff1','feaSelector1','-v7.3');
end

dir = '/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch';
w=load(strcat(dir,'/map.mat'));
map=w.map;

nLetters_range = 10;
nSeq_range = 1:10;

%skip those bad subjects, as well as Greek subjects
invalid=[1,3,6,26,41,48,51,54,58];
greek=[70:71,73:74,76:81];
subj_list=[invalid, greek];
%specify which subject
zs=1:length(trainData1);
inds=setdiff(zs, subj_list);
cv=setdiff(zs, invalid);

%feaSelector contains indices, coeff contains values
%translating feaSelector and coeff into weights (sparse)
ws=zeros(length(trainData1),size(trainData1{1},2));     
for i=cv
    ws(i,feaSelector1{i})=coeff1{i};
end

%set 'sp' symbol to '_' to avoid errors
parameters=parameters1{2,1};
parameters.TargetDefinitions.Value{36,1}='_';
targets = lower(cell2mat(parameters.TargetDefinitions.Value(:,1)));

% load transition matrix or compile one if not exist yet
sprintf('compiling transition matrix')
try
    load('/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/transition_matrix.mat')
catch e    
    transition_matrix=zeros(length(targets)*length(targets));
    lambda=1; % add lambda smoothing
    for i=1:length(targets)
        for j=1:length(targets)
            for k=1:length(targets)
                denom=length(targets)*lambda;
                try
                    denom=map.(['t' targets([i j])' 'X'])+length(targets)*lambda;
                catch e
                end
                try
                    denom=denom-map.(['t' targets([i j])' '0']);
                catch e
                end
                num=lambda;
                try
                    num=map.(['t' targets([i j k])'])+lambda;
                catch e
                end
                transition_matrix((i-1)*length(targets)+j,(j-1)*length(targets)+k)=num/denom;
            end
        end
    end
    save('/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/transition_matrix.mat', 'transition_matrix')
end

blank=zeros(1,length(targets)*length(targets));
blank(end)=1;

answers =cell(length(inds),1);
all_coeffs=cell(length(inds),1);
all_feaSelectors=cell(length(inds),1);
all_mean_as=cell(length(inds),1);
all_mean_ns=cell(length(inds),1);
all_std_as=cell(length(inds),1);
all_std_ns=cell(length(inds),1);
all_results=cell(length(inds),1);
all_accs=cell(length(inds),1);
all_trells=cell(length(inds),1);
all_labels=cell(length(inds),1);


enable_parallel = false;
sequential_range = 1:9;

if enable_parallel
    futures = parallel.FevalFuture.empty(length(inds), 0);
    start=1;
    while start+1<=length(inds)
        for i=start:start+1
            futures(i) = parfeval(backgroundPool, @P300_baum_welch, 11, inds(i), nLetters_range, nSeq_range, trainData1, allStim1, parameters1, labels1, [], ws, parameters, cv, blank, transition_matrix);
        end

        for j=start:start+1
            [answer, results, accs, coeffs, feaSelectors, mean_as, mean_ns, std_as, std_ns, trells, labs] = fetchOutputs(futures(j));
        
            answers{j}=answer;
            all_results{j}=results;
            all_accs{j}=accs;
            all_coeffs{j}=coeffs;
            all_feaSelectors{j}=feaSelectors;
            all_mean_as{j}=mean_as;
            all_mean_ns{j}=mean_ns;
            all_std_as{j}=std_as;
            all_std_ns{j}=std_ns; 
            all_trells{j}=trells;
            all_labels{j}=labs;
        end

        start=start+2;
        save("/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/results/all_accs.mat", "all_accs");
    end
else
    for i=sequential_range
        [answer, results, accs, coeffs, feaSelectors, mean_as, mean_ns, std_as, std_ns, trells, labs] = P300_baum_welch(inds(i), nLetters_range, nSeq_range, trainData1, allStim1, parameters1, labels1, [], ws, parameters, cv, blank, transition_matrix);
       
        answers{i}=answer;
        all_results{i}=results;
        all_accs{i}=accs;
        all_coeffs{i}=coeffs;
        all_feaSelectors{i}=feaSelectors;
        all_mean_as{i}=mean_as;
        all_mean_ns{i}=mean_ns;
        all_std_as{i}=std_as;
        all_std_ns{i}=std_ns; 
        all_trells{i}=trells;
        all_labels{i}=labs;
    end
    save("/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/results/all_accs.mat", "all_accs");
end


