function [W, A, B, output, error] = train_rbm(Epoch, numV, numH, data, eW, eV, eH, wc, initialM, finalM)
% train rbm
% variables
[sampleNum, dims, batchNum] = size(data);
W = randn(numV, numH) * 0.1; % weight matrix between visible noeds and hidden nodes
A = zeros(1, numV); % bias of visible nodes
B = zeros(1, numH); % bias of hidden nodes
deltaW = zeros(numV, numH); % store gradient of W (same below)
deltaA = zeros(1, numV);
deltaB = zeros(1, numH);
output = zeros(sampleNum, numH, batchNum); % store output of hidden nodes of every sample
p_h1_v = zeros(sampleNum,numH); % p(h=1|v)
newH = zeros(sampleNum,numH);  % hidden nodes after update
gradForward = zeros(numV,numH);  % forward gradient
gradBack    = zeros(numV,numH);  % backward gradient
error = zeros(1, Epoch);

% training begin
for epoch=1:Epoch
    for batch=1:batchNum
        if mod(batch,150)==0
            fprintf('epoch %d batch %d\r', epoch, batch);
        end
        samples = data(:, :, batch);
        % calculate p(h=1|v) and sample h
        p_h1_v = 1 ./ (1 + exp(-samples*W - repmat(B, sampleNum, 1)));
        output(:, :, batch) = p_h1_v;
        h_sampled = p_h1_v > rand(sampleNum, numH); % Gibbs sampling
        % calculate forward gradient
        gradForward = samples' * p_h1_v;
        % calculate backward gradient
        p_v1_h = 1 ./ (1 + exp(-h_sampled*W' - repmat(A, sampleNum, 1)));
        newH = 1 ./ (1 + exp(-p_v1_h*W - repmat(B, sampleNum, 1)));
        gradBack = p_v1_h' * newH;
        % update params
        error(epoch) = error(epoch) + sum(sum((samples - p_v1_h) .^ 2));
        if epoch > 5
            momentum = finalM;
        else
            momentum = initialM;
        end
        deltaW = momentum*deltaW + eW*((gradForward - gradBack)/sampleNum - wc*W);
        deltaA = momentum*A + (eV/sampleNum)*(sum(samples) - sum(p_v1_h));
        deltaB = momentum*B + (eH/sampleNum)*(sum(p_h1_v) - sum(newH));
        W = W + deltaW;
        A = A + deltaA;
        B = B + deltaB;
    end
    fprintf('epoch %d error: %6.1f \n', epoch, error(epoch)/batchNum);
    error(epoch) = error(epoch) / batchNum;
end
end

