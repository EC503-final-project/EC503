Epoch = 10;
numV = 28*28;
numH = 500;
eW = 0.1;
eV = 0.1;
eH = 0.1;
wc = 0.0002;
initialM = 0.5;
finalM = 0.9;
gsampleNum = 1;
[train_data, test_data] = make_datas();

% [W, A, B, input, train_error] = train_rbm2(Epoch, numV, numH, train_data, eW, eV, eH, wc, initialM, finalM);
% plot(1:Epoch, train_error);
% title("training error of RBM");
% save('W.mat', 'W');
% save('A.mat', 'A');
% save('B.mat', 'B');

A = matfile('A.mat');
A = A.A;
B = matfile('B.mat');
B = B.B;
W = matfile('W.mat');
W = W.W;
error = reconstruct(A, B, W, test_data(:, :, gsampleNum), numV, numH);
% random_reconstruct(A, B, W, test_data(:, :, gsampleNum), numV, numH);