format compact;
clear;
close all; 

addpath(genpath(['\funs']));

load('yale.mat');

X = mapminmax(X',-1,1); %%% standardization
X = X';
c=length(unique(y));


nNeuron = 1000;
islocal = 0;

nNN = 3;
dE = 4;
rPara = 100;

for ind_repeat = 1:10
    try
        [Xp, W, la, A, evs, error] = ELMCLR(X', c, dE, nNN,-1,islocal,nNeuron,rPara);
    catch
        error = 2;
    end
    if error ~=0
        temp0(ind_repeat,:) = [0 0 0];
    else
        temp0(ind_repeat,:) = ClusteringMeasure(y, la);
    end
end
acc.mean = mean(temp0(:,1));
acc.std = std(temp0(:,1),1)

nmi.mean = mean(temp0(:,2));
nmi.std = std(temp0(:,2),1)


filename = ['result_elmclr_yale']
save(filename);
