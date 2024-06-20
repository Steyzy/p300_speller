function [all_files, trainData1, allStim1, parameters1, labels1, coeff1, feaSelector1] = P300_train(szFile)


if ischar(szFile)
    dir = szFile;
    subjects_temp = ls(strcat(szFile,'p300*')); %Subject Files must start with 'p300'

    subjects = cell(size(subjects_temp,1),1);
    for i=1:size(subjects_temp,1) %Creates a list of the file locations of the subjects
        subjects{i}=strtrim(strcat(dir,subjects_temp(i,:)));
    end;

    files = cell(size(subjects,1),1);
    all_files = cell(size(subjects,1),1); %list of all data files separated by subject

    for i = 1:length(subjects)
        files{i} = ls(char(strcat(subjects(i),'\*.dat'))); %Creates a list of data files within a particular subject
        all_files{i} = strtrim(strcat(subjects{i},'\',files{i,:})); %puts each list of data files into a list
    end;      
end;


DSFactor = 12;
downsampleFunc =@ DownSampleMatByAvg;
nAvgTrial = 1;
epochLength = 600;

for z=1; 
%for z=1:length(all_files) %parsing through subjects and findig a matrix for each
    
    allData   = [];
    allStim   = [];
    allLabels = [];
    
    for j=1:size(all_files{z},1)
        sprintf('Training file %d for subject %d',j,z)
        [ signal, states, parameters ] = load_bcidat(all_files{z}(j,:));
        parameters1{z,j}=parameters;
        rate = parameters.SamplingRate.NumericValue;
        epochPoints = ceil(epochLength * rate / 1000);
        
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
                %tempData=[tempData, DownSample4Feature(signal(onsetIndices(i):(onsetIndices(i)+epochPoints-1),chan),DSFactor,nAvgTrial,[],downsampleFunc)];
                tempData=[tempData, signal(onsetIndices(i):(onsetIndices(i)+epochPoints-1),chan)];
            end;
            tempData2=[tempData2;tempData];
        end;
        allData=[allData,tempData2];
    end;

     [coeff,  feaSelector] = BuildStepwiseLDA(allData', allLabels);
     coeff1{z}=coeff;
     feaSelector1{z}=feaSelector;
     trainData1{z}=allData'; %?????
     labels1{z}=allLabels; %?????
     allStim1{z}=allStim;
end;
