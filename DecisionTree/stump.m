clear;
X = csvread('train.csv',1,0,[1 0 500 4]);
Y = csvread('train.csv',1,5);

%%
mdl = fitctree(X,Y,'Surrogate','on','MaxNumSplits',1,'CrossVal','on');
%% 

view(mdl.Trained{1},'Mode','graph')