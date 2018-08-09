% HW04, Problem 2 (learn f=1/x)

% function to train 2-layer neural net
% Inputs:
% train_features - training samples
% train_labels - training labels
% test_features - test samples
% test_labels - test labels
% num_hidden - number of hidden PEs
% gamma - learning rate
% momentum - momentum
% iter - number of epochs
% batch_size - epoch size
% tol - early stopping condition
function [train_score, test_score] = neuralNetwork(train_features, train_labels, test_features, test_labels, num_hidden, gamma, momentum, iter, batch_size, tol)
    % get number of input and output nodes
    num_input = size(train_features, 2);
    num_output = size(train_labels, 2);
    
    num_train = size(train_features, 1);
    train_bias = ones(num_train, 1);
    train_data = [train_bias train_features];
    
    num_test = size(test_features, 1);
    test_bias = ones(num_test, 1);
    test_data = [test_bias test_features];

    % Initialize Weights
    weights_1 = rand(num_input+1,num_hidden).*0.2-0.1;
    weights_2 = rand(num_hidden+1,num_output).*0.2-0.1;
    
    % initialize previous delta terms for momentum
    prev_delta_1 = 0;
    prev_delta_2 = 0;
 
    num_batches = num_train / batch_size;
    
    learn_steps = [];
    train_err = [];
    test_err = [];
    % Train over iter epochs
    for i = 1:iter
        % compute mini batches
        batch_start = mod(i-1,num_batches)*batch_size+1;
        batch_end = mod(i-1,num_batches+1)*batch_size;
        
        x = train_data(batch_start:batch_end,:);
        y = train_labels(batch_start:batch_end,:); 
        b = train_bias(batch_start:batch_end,:);
        
        % Forward propagation
        z_1 = x * weights_1; % n-by-10
        a_1 = [b tanh(z_1)]; % n-by-10
        z_2 = a_1 * weights_2; % n-by-1
        y_hat = tanh(z_2); % n-by-1

        % Check error convergence
        if recallScore(y_hat, y) < tol
            break;
        end        
     
        % Collect error at every 100 epochs
        if mod(i, 100) == 0
            learn_steps = [learn_steps i*batch_size/100000];
            
            [train_pred, test_pred] = recall(train_data, test_data, weights_1, weights_2);
            train_score = recallScore(train_pred, train_labels);
            train_err = [train_err train_score];
            test_score = recallScore(test_pred, test_labels);
            test_err = [test_err test_score];
        end
            
        % Backward propagation - update weights
        delta_2 = gamma .* a_1.' * (tanhPrime(z_2) .* (y - y_hat));
        delta_1 = gamma .* x.' * ((((y - y_hat) .* tanhPrime(z_2)) * weights_2(2:end,:).') .* tanhPrime(z_1));
        weights_2 = weights_2 + delta_2 + (momentum * prev_delta_2); % 10-by-1
        weights_1 = weights_1 + delta_1 + (momentum * prev_delta_1); % 2-by-10
        prev_delta_2 = delta_2;
        prev_delta_1 = delta_1;
    end
    
    [train_pred, test_pred] = recall(train_data, test_data, weights_1, weights_2);
    train_score = recallScore(train_pred, train_labels);
    test_score = recallScore(test_pred, test_labels);
    
    % for iris data
       [target_val, target_arg] = max(test_labels.');
 [pred_val, pred_arg] = max(test_pred.');
      sum(target_arg == pred_arg) / 75
      plotClassification(target_arg, pred_arg);
    
    % for 1/x data
%     figure(1);
%     plotError(learn_steps, train_err, test_err);
%     hold on;
%     figure(2);
%     plotValues(train_features, train_pred, train_labels, test_features, test_pred, test_labels);
%     hold on;
%     test_err(length(test_err))
end

% predicts training and test data with given weights
function [train_pred, test_pred] = recall(train_data, test_data, weights_1, weights_2)
    num_train = size(train_data, 1);
    train_bias = ones(num_train, 1);
    train_pred = tanh([train_bias tanh(train_data * weights_1)] * weights_2);
    
    num_test = size(test_data, 1);
    test_bias = ones(num_test, 1);
    test_pred = tanh([test_bias tanh(test_data * weights_1)] * weights_2);
end

% mean squared error
function score = recallScore(pred, labels)
    score = 1 / length(pred) * sum((pred - labels).^2);
end

% derivative of tanh
function gradient = tanhPrime(x)
    gradient = 1 - tanh(x).^2;
end

% plot training and test error
function plotError(learn_steps, train_err, test_err)
    plot(learn_steps, train_err, learn_steps, test_err);
    title({'Training and Test Error by Number of Learning Steps', '(recorded every 10k learning steps)'})
    xlabel('3M Learning Steps');
    ylabel('Mean Squared Error');
    legend('train error', 'test error');
end

% plot actual and desired values
function plotValues(train_features, train_pred, train_labels, test_features, test_pred, test_labels,style)
    scatter(train_features, train_pred, 5,  'filled');
    hold on;
    scatter(test_features, test_pred, 5,  'filled');
    scatter([train_features; test_features], [train_labels; test_labels], 5,  'filled')
    title({'Target and Actual Outputs', 'after 3M Learning Steps'});
    xlabel('x');
    ylabel('y=f(x)');
    legend('Training Recall', 'Test Recall', 'Target (1/x)');    
end

% plot iris classiifcation results
function plotClassification(test_labels, test_pred)
    plot(1:length(test_labels),test_pred,'r.')
    hold on;
    plot(1:length(test_labels),test_labels,'bo')
    title({'Desired and Predicted Classification Labels', 'after 750k Learning Steps'});
    ylabel('Classification');
    legend('Predicted','Desired');
end