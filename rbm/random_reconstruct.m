function [] = random_reconstruct(A, B, W, samples, numV, numH)
% reconstruct image using RBM
input = rand(10, numH);
input(input < 0.5) = 0;
input(input >= 0.5) = 1;
p_h1_v = 1./(1 + exp(-input*W' - repmat(A,10,1)));
recons = p_h1_v > rand(10, numV); % Gibbs sampling
imgs = reshape(recons, 10, 28, 28);
for i=1:10
    imwrite(reshape(imgs(i, :, :), 28, 28), strcat('./img/random', int2str(i), '.jpg'));
end
end