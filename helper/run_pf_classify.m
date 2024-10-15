bci_file='U:\Data\P300 data\p300_wc\wc_subject_1\PF32_2016_04_01FF001\PF32_2016_04_01FFS001R01.dat';
corpus_dir='U:\Data\brown\';
thresh=.999;
nPart=10000;
epochLength=600;

[signal,states,parameters]=load_bcidat(bci_file);
rate=parameters.SamplingRate.NumericValue;
target_word=strrep(lower(parameters.TextToSpell.Value{1}),' ','_');
stimuli=states.StimulusCode;
type=states.StimulusType;
inSequence=states.PhaseInSequence;
nRows=parameters.NumMatrixColumns.NumericValue;
nCols=parameters.NumMatrixRows.NumericValue;
targetDefinitions=parameters.TargetDefinitions.Value(:,1);
for k=1:length(targetDefinitions)
    if(strcmp(targetDefinitions{k},'sp'))
        targetDefinitions{k}='_';
    end;
    if(length(targetDefinitions{k})>1)
        targetDefinitions{k}='0';
    end;
end;
targets = lower(cell2mat(targetDefinitions));

[output,nStims]=pf_classify(signal,stimuli,type,inSequence,target_word,rate,epochLength,thresh,nRows,nCols,targets,corpus_dir,nPart);

[output.keys' output.values']

nStims