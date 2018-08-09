function oneOverXTest()
    % Initialize data
    % shuffle evenly spaced data to not have duplicate x values
    even_x = linspace(0.1,1,300);
    shuffled_x = even_x(randperm(300));

    train_features = shuffled_x(1:200).';
    train_labels = 1./train_features./10; % Scale labels to [-1, 1]

    test_features = shuffled_x(201:end).';
    test_labels = 1./test_features./10;

%     test_features = [test_features test_features];
%     test_labels = [test_labels test_labels];
    % Network parameters
    gamma = 0.005;
    iter = 30000;
    tol = 0.0001;
    momentum = 0.1;
    batch_size = 200;
    num_hidden = 10;
   [train_score, test_score] = neuralNetwork(train_features, train_labels, test_features, test_labels, num_hidden, gamma, momentum, iter, batch_size, tol)
end