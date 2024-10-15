addpath('helper');
szFile = '/Users/yangziyi/Desktop/Neuro Research/p300_data/';

% % extract data from 83 subjects, each subject could have more than 1 trial
% if ischar(szFile)
%     subjects_temp = {dir(strcat(szFile,'subject*')).name}';
% 
%     subjects = cell(size(subjects_temp,1),1);
%     for i=1:size(subjects_temp,1) %Creates a list of the file locations of the subjects
%         subjects{i}=strtrim(strcat(szFile,subjects_temp(i,:)));
%     end;
% 
%     files = cell(size(subjects,1),1);
%     all_files = cell(size(subjects,1),1); %list of all data files separated by subject
% 
%     for i = 1:length(subjects)
%         files{i} = {dir(char(strcat(subjects{i},'/*.dat'))).name}'; % contains file name of all dat files
%         all_files{i} = strtrim(strcat(subjects{i},'/',files{i,:})); % contains full path of all data files
%     end;     
% end;
% 
% numFiles = cellfun(@(x) size(x, 1), all_files);

try
    load('/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/all_subjects.mat');
catch e
    [all_files, trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1] = P300_train(szFile);
    save('/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch/all_subjects.mat','all_files','trainData1','allStim1','parameters1','labels1','coeff1','feaSelector1','-v7.3');
end

dir = '/Users/yangziyi/Desktop/Neuro Research/p300_baum_welch';
w=load(strcat(dir,'/map.mat'));
map=w.map;

[answers, all_results, all_accs, all_coeffs, all_feaSelectors, all_mean_as, all_mean_ns, all_std_as, all_std_ns,all_channels,all_trells,all_labels] = P300_baum_welch(trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1, map,1,[]);


% ws0=zeros(length(trainData1),size(trainData1{1},2)); 
% for i=1:size(ws0,1)
%     ws0(i,feaSelector1{i})=coeff1{i};
% end;
% % temp=[];
% % temp(:,:)=sum(reshape(ws0.^2,[15,13,32]),2);
% % temp2=temp./(sum(temp,2)*ones(1,32));
% % [h,p]=ttest(temp2,[],.05)
% 
% coeff2=coeff1;
% feaSelector2=feaSelector1;
% for i=1:length(coeff2)
%     coeff2{i}(:)=0;
%     feaSelector2{i}(:)=1;
%     %coeff2{i}(:)=normrnd(zeros(length(coeff2{i}),1),1);
%     %feaSelector2{i}(:)=ceil(rand(length(feaSelector2{i}),1)*32);
% end;
% 

% 
% [answers, all_results, all_accs, all_coeffs, all_feaSelectors, all_mean_as, all_mean_ns, all_std_as, all_std_ns,all_channels,all_trells,all_labels] = P300_baum_welch(trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1, map,1,[]);
% 
% % ws1=zeros(size(ws0));
% % for i=1:size(ws1,1)
% %     ws1(i,all_feaSelectors{i}{10})=all_coeffs{i}{10};
% % end;
% % temp=[];
% % temp(:,:)=sum(reshape(ws1.^2,[15,13,32]),2);
% % temp2=temp./(sum(temp,2)*ones(1,32));
% % [h,p]=ttest(temp2,[],.05)
% % 
% % x=cell2mat(all_accs{15})'
% % 
% % szFiles={'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjA\long\',...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjB\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjC\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjD\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjE\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjF\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjG\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjH\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjI\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjJ\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjK\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjL\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjM\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjN\long\'...
% %     'Y:\users\wspeier\Data\P300 data\p300_jessica\p300subjO\long\'};
% % answers=cell(size(szFiles));
% % staticResults1=cell(size(szFiles));
% % staticResults2=cell(size(szFiles));
% % NLPResults1=cell(size(szFiles));
% % NLPResults2=cell(size(szFiles));
% % NLP2Results1=cell(size(szFiles));
% % NLP2Results2=cell(size(szFiles));
% % for i=1:length(szFiles)
% %     ['file:' szFiles{i}]
% %     %[answers{i},staticResults1{i},staticResults2{i},NLPResults1{i},NLPResults2{i},NLP2Results1{i},NLP2Results2{i},attScores,adjScores,distScores]=P300_HMM(szFiles{i},map);
% %     %[answers{i},staticResults1{i},staticResults2{i},NLPResults1{i},NLPResults2{i},NLP2Results1{i},NLP2Results2{i},attScores,adjScores,distScores]=P300_HMM(szFiles{i});
% %     %[answers{i},staticResults1{i},staticResults2{i},NLPResults1{i},NLPResults2{i},NLP2Results1{i},NLP2Results2{i}]=P300_HMM3(szFiles{i},map);
% %     %[answers{i},staticResults1{i},staticResults2{i},NLPResults1{i},NLPResults2{i},NLP2Results1{i},NLP2Results2{i}]=P300_HMM3(szFiles{i});
% %     %[nanswers{i},nstaticResults1{i},nstaticResults2{i},nNLPResults1{i},nNLPResults2{i},nNLP2Results1{i},nNLP2Results2{i}]=P300_HMM3(szFiles{i},map);
% %     %[answers{i},staticResults1{i},NLPResults1{i}]=P300_baum_welch_test_1(szFiles{i},map,15,all_coeffs{i}{10},all_feaSelectors{i}{10});
% %     [answers{i},staticResults2{i},NLPResults2{i}]=P300_baum_welch_test_2(szFiles{i},trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1,i,map,15);
% % end;
% 
