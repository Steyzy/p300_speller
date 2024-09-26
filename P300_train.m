function [all_files, trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1] = P300_train(szFile)


% extract data from 83 subjects, each subject could have more than 1 trial
if ischar(szFile)
    subjects_temp = {dir(strcat(szFile,'subject*')).name}';

    subjects = cell(size(subjects_temp,1),1);
    for i=1:size(subjects_temp,1) %Creates a list of the file locations of the subjects
        subjects{i}=strtrim(strcat(szFile,subjects_temp(i,:)));
    end;
    
    files = cell(size(subjects,1),1);
    all_files = cell(size(subjects,1),1); %list of all data files separated by subject
    
    for i = 1:length(subjects)
        files{i} = {dir(char(strcat(subjects{i},'/*.dat'))).name}'; % contains file name of all dat files
        all_files{i} = strtrim(strcat(subjects{i},'/',files{i,:})); % contains full path of all data files
    end;     
end;


DSFactor = 12;
downsampleFunc =@ DownSampleMatByAvg;
nAvgTrial = 1;
epochLength = 600;

% Create a parallel pool
pc = parcluster('local');
pc.NumWorkers = 10; % Adjust the number of workers as needed
parpool(pc);

parfor z=1:length(all_files) %parsing through subjects and findig a matrix for each

    allData   = [];
    allStim   = [];
    allLabels = [];

    bad_channels = [];
    %parse the excel file and set bad channels to 0
    subj_name = subjects_temp{z};
    [~, ~, raw] = xlsread("bad_channels.xlsx");
    raw = raw(2:end,:);
    for i=1:size(raw, 1)
        if(raw{i, 1}==subj_name)
            for j=2:size(raw, 2)
                if(~isnan(raw{i, j}))
                    bad_channels = [bad_channels, raw{i, j}]
                end
            end
        end
    end

    %for all files of subject z
    for j=1:4
        out_of_range = false;
        try
            all_files{z}{j};
        catch e
            out_of_range = true;
        end

        if out_of_range
            break
        end

        sprintf('Training file %d for subject %d',j,z)
        [ signal, states, parameters ] = load_bcidat(all_files{z}{j});
        parameters1{z,j}=parameters;
        rate = parameters.SamplingRate.NumericValue;
        [num,den]=butter(1,1/(rate/2),'high'); %high pass the signal
        signal=filter(num,den,double(signal));
        epochPoints = ceil(epochLength * rate / 1000);

        stimulusType = states.StimulusType;
        stimulusCode = double(states.StimulusCode);
        %keep only the onset points, ignoring the flat regions
        stimulusCode(1:rate)=0;     
        stimulusCode(stimulusCode==circshift(stimulusCode,1))=0;
        
        onsetIndices = find(stimulusCode>0);    %time points when stimuli start
        allLabels = [allLabels;double(stimulusType(onsetIndices))];
        allStim  = [allStim;stimulusCode(onsetIndices)];
        tempData2 = [];
        
        %iterate over channels
        for chan = 1:size(signal,2)
            tempData = [];
            for i=1:length(onsetIndices)
                %tempData=[tempData, DownSample4Feature(signal(onsetIndices(i):(onsetIndices(i)+epochPoints-1),chan),DSFactor,nAvgTrial,[],downsampleFunc)];
                tempData=[tempData, signal(onsetIndices(i):(onsetIndices(i)+epochPoints-1),chan)];
            end;
            %set tempData to all 0's if is bad channel
            if(ismember(chan, bad_channels))
                tempData2 = [tempData2; zeros(size(tempData))];
            else
                tempData2=[tempData2; tempData];
            end;
        end;
        allData=[allData,tempData2];
    end;

     [coeff, feaSelector] = BuildStepwiseLDA(allData', allLabels);
     coeff1{z}=coeff;
     feaSelector1{z}=feaSelector;
     trainData1{z}=allData'; %?????
     labels1{z}=allLabels; %?????
     allStim1{z}=allStim;
end;

% Close the parallel pool
delete(gcp('nocreate'));
