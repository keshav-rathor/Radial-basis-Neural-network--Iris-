close all; 
clear all;
clc
%% load divided input data set
load fisheriris
% coding (+1/-1) of 3 classes
a = [-1 -1 +1]';
b = [-1 +1 -1]';
c = [+1 -1 -1]';
% define training inputs
rand_ind = randperm(50);
trainSeto = meas(rand_ind(1:35),:);
trainSeto=trainSeto';
trainVers = meas(50 + rand_ind(1:35),:);
trainVers=trainVers';
trainVirg = meas(100 + rand_ind(1:35),:);
trainVirg=trainVirg';
trainInp = [trainSeto trainVers trainVirg];
% define targets
tmp1 = repmat(a,1,length(trainSeto));
tmp2 = repmat(b,1,length(trainVers));
tmp3 = repmat(c,1,length(trainVirg));
T = [tmp1 tmp2 tmp3];

%% choose a spread constant (1st step)
spread = 2.1;
Cor = zeros(2,209);
Sp = zeros(1,209);
Sp(1,1) = spread;
for i = 1:209,
spread = spread - 0.01;
Sp(1,i) = spread;
% choose max number of neurons
K = 40;
% performance goal (SSE)
goal = 0;
% number of neurons to add between displays
Ki = 5;
% create a neural network
net = newrb(trainInp,T,goal,spread,K,Ki);
% simulate RBFN on training data
Y = sim(net,trainInp);
% define validation vector
rand_ind = randperm(50);
valSeto = meas(rand_ind(1:20),:);
valSeto= valSeto';
valVers = meas(50 + rand_ind(1:20),:);
valVers=valVers';
valVirg = meas(100 + rand_ind(1:20),:);
valVirg=valVirg';

valInp = [valSeto valVers valVirg];
tmp1 = repmat(a,1,length(valSeto));
tmp2 = repmat(b,1,length(valVers));
tmp3 = repmat(c,1,length(valVirg));
valT = [tmp1 tmp2 tmp3];
[Yval,Pf,Af,E,perf] = sim(net,valInp,[],[],valT);
% calculate [%] of correct classifications
Cor(1,i) = 100 * length(find(T.*Y > 0)) / length(T);
Cor(2,i) = 100 * length(find(valT.*Yval > 0)) / length(valT);
end
figure
pl = plot(Sp,Cor/3);
set(pl,{'linewidth'},{1,3}');
%% choose a spread constant (2nd step)
spread = 1.0;
Cor = zeros(2,410);
Sp = zeros(1,410);
Sp(1,1) = spread;
for i = 1:410,
spread = spread - 0.001;
Sp(1,i) = spread;
% choose max number of neurons
K = 40;
% performance goal (SSE)
goal = 0;
% number of neurons to add between displays
Ki = 5;
% create a neural network
net = newrb(trainInp,T,goal,spread,K,Ki);
% simulate RBFN on training data
Y = sim(net,trainInp);
% define validation vector
valInp = [valSeto valVers valVirg];
tmp1 = repmat(a,1,length(valSeto));
tmp2 = repmat(b,1,length(valVers));
tmp3 = repmat(c,1,length(valVirg));
valT = [tmp1 tmp2 tmp3];

[Yval,Pf,Af,E,perf] = sim(net,valInp,[],[],valT);
% calculate [%] of correct classifications
Cor(1,i) = 100 * length(find(T.*Y > 0)) / length(T);
Cor(2,i) = 100 * length(find(valT.*Yval > 0)) / length(valT);
end
figure
pl = plot(Sp,Cor/3);
set(pl,{'linewidth'},{1,3}');
%% final training
spr = 0.8;
fintrain = [trainInp valInp];
finT = [T valT];
net = newrb(fintrain,finT,goal,spr,K,Ki);
% simulate RBFN on training data
finY = sim(net,fintrain);
% calculate [%] of correct classifications
finCor = 100 * length(find(finT.*finY > 0)) / length(finT);
fprintf('\nSpread = %.3f\n',spr)
fprintf('Num of neurons = %d\n',net.layers{1}.size)
fprintf('Correct class = %.3f %%\n',finCor/3)
% plot targets and network response
figure;
plot(T')
ylim([-2 2])
set(gca,'ytick',[-2 0 2])
hold on
grid on
plot(Y','r')
legend('Targets','Network response')
xlabel('Sample No.')
%% Testing
rand_ind = randperm(50);
testSeto = meas(rand_ind(36:50),:);
testSeto=testSeto';
testVers = meas(50 + rand_ind(36:50),:);
testVers=testVers';
testVirg = meas(100 + rand_ind(36:50),:);
testVirg=testVirg';
% define test set
testInp = [testSeto testVers testVirg];

temp1=repmat(a,1,length(testSeto));
temp2=repmat(b,1,length(testVers));
temp3=repmat(c,1,length(testVirg));

testT = [temp1 temp2 temp3];
testOut = sim(net,testInp);
testCor = 100 * length(find(testT.*testOut > 0)) / length(testT);
fprintf('\nSpread = %.3f\n',spr)
fprintf('Num of neurons = %d\n',net.layers{1}.size)
fprintf('Correct class = %.3f %%\n',testCor/3)
% plot targets and network response
figure;
plot(testT')
ylim([-2 2])
set(gca,'ytick',[-2 0 2])
hold on
grid on
plot(testOut','r')
legend('Targets','Network response')
xlabel('Sample No.')