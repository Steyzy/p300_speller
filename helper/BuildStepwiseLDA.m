function  [coeff,  feaSelector] = BuildStepwiseLDA(feature, classLabel,allowEmpty)
%
%
%
%    Build a stepwise linear discriminant analysis classifier

if nargin < 3
    allowEmpty = 1;
end;

MaxFeatureNumber = 60;%size(feature, 2) * 0.5;
%MaxFeatureNumber = 160;
%MaxFeatureNumber = size(feature,2)/10;
%pEnter = 0.1;
%pRemove = 0.15;

%feature(:,1:5760)=0;
%feature(:,6961:end)=0;

%[B, SE, PVAL,in] = stepwisefit(feature, classLabel,'maxiter',MaxFeatureNumber,'display','off','penter',pEnter,'premove',pRemove);
coeff=[];
feaSelector=[];
pEnter = 0.1;%0.05;
pRemove = 0.15;%0.35;
counter = 0;
while(isempty(feaSelector))
    counter = counter + 1;
    [B, SE, PVAL,in] = stepwisefit(feature, classLabel,'display','off','penter',pEnter,'premove',pRemove,'maxiter',MaxFeatureNumber);
    
    feaSelector = find(in ~= 0);
    coeff = B(feaSelector);
    pEnter = pEnter + 0.05;
    pRemove = pRemove + 0.05;
    if and(isempty(feaSelector),counter>4)
        feaSelector = find(in==0);
        coeff=B(feaSelector);
    end;
    if or(allowEmpty,~isempty(feaSelector))
        break;
    else
        sprintf('no features for attempt %d',counter)
    end;
end;