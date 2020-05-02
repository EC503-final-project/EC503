function [error] = reconstruct(A, B, W, samples, numV, numH)
% reconstruct image using RBM
[sampleNum, dims, batchNum] = size(samples);
for i=1:sampleNum
    samples(i, :, 1) = imnoise(samples(i, :, 1), 'gaussian');
end
input = [samples(:, :, 1) ones(sampleNum, 1)];
wForward = [W; B];
wBack = [W'; A];
tmp = 1./(1 + exp(-input*wForward));
tmp = [tmp  ones(sampleNum,1)];
recons = 1./(1 + exp(-tmp*wBack));
error = sum(sum((samples - recons) .^ 2));
imgs = reshape(recons, sampleNum, 28, 28);
for i=1:10
    imwrite(reshape(samples(i, :, 1), 28, 28), strcat('./img/ori', int2str(i), '.jpg'));
    imwrite(reshape(imgs(i, :, :), 28, 28), strcat('./img/', int2str(i), '.jpg'));
end
end