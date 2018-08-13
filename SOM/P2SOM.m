function P2SOM
clear;
%% Initialize input data
% variance = 0.1, centered at (7, 7, 0)
X1=sqrt(0.1)*randn(3,1000);
X1=detrend(X1')';
X1(1,:)=X1(1,:)+7*ones(1,1000);
X1(2,:)=X1(2,:)+7*ones(1,1000);
X2=sqrt(0.1)*randn(3,1000);
X2=detrend(X2')';
X2(1,:)=X2(1,:)+7*ones(1,1000);
X3=sqrt(0.1)*randn(3,1000);
X3=detrend(X3')';
X3(2,:)=X3(2,:)+7*ones(1,1000);
X4=sqrt(0.1)*randn(3,1000);
X4=detrend(X4')';
X=[X1 X2 X3 X4];
X=X(:,randperm(length(X)));
%% Build network
% PEs on input layer
Inum=4000; 

% Kohonen PE initialization
Kohonen_rows = 10;
Kohonen_cols = 10;
topo = [Kohonen_rows, Kohonen_cols]; 
grid_mat = zeros(topo(1),topo(2));

% plot threshold
dist_threshold = 0.5;

% Weights initialization
weight = rand(3,Kohonen_rows,Kohonen_cols);
weight(1,:,:)=reshape(weight(1,:,:),[],10);

% convergence criteria
converge = 100;
threshold =  0.01;

% Learning rate
gamma=1;   
gamma_min=0.1;

% Distance
sigma_max = 5;
sigma_min = 1.01;

% Steps
t = 0;
total_steps = 00;

%% Main program
while and(t<=total_steps, converge>threshold)
    if mod(t,500) == 0
        gamma = gamma - 0.05;
    end
    if gamma <= gamma_min
        gamma = gamma_min;
    end
    sigma = sigma_max * (1-t/total_steps);
    if sigma <= sigma_min
        sigma = sigma_min;
    end
    [input_index, best_row, best_col, mindist] = calculate_best(X, weight,Inum);
    conv = 0;
    for i = 1:10
        for j = 1:10
            neighbour = exp(-abs((i-best_row).^2+(j-best_col).^2)/(sigma.^2));
            update = gamma * neighbour * (X(:,input_index)-weight(:,i,j));
            weight(:,i,j) = weight(:,i,j) + update;
            conv = conv + abs(update);
        end
    end
    grid_mat(best_row, best_col)= grid_mat(best_row, best_col)+1;
    if conv < converge
        converge = conv;
    end
    w1=reshape(weight(1,:,:),[],10);
    w2=reshape(weight(2,:,:),[],10);
    w3=reshape(weight(3,:,:),[],10);
    figure(1)
    plot3(X(1,:),X(2,:),X(3,:),'.b')
    hold on
    plot3(w1,w2,w3,'or')
    plot3(w1,w2,w3,'k','linewidth',2)
    plot3(w1',w2',w3','k','linewidth',2)
    hold off
    title(['t=' num2str(t)]);
    drawnow
    t = t+1;
end
gridmap(grid_mat,total_steps)
% visualUmatrix(0,w1,w2,w3)
dist_fence = grid_plot(weight,topo,dist_threshold);
writeFile(dist_fence);

%% Calculate best PEs
function [input_index, best_row, best_col, mindist] = calculate_best(input, weight,Inum)
    input_index = unidrnd(Inum);
    mindist = 100;
    for i = 1:10
        for j = 1:10
            temp1 = sum(((input(:,input_index)-weight(:,i,j)).^2).^0.5);
            if temp1 < mindist
                best_row = i;
                best_col = j;
                mindist = temp1;
            end
        end
    end
end

%% plot the SOM lattic
function dist_fence = grid_plot(weights,topo,dist_threshold)
dist_fence = zeros(2*topo(1)-1,2*topo(2)-1);
for i = 1:topo(1) 
    for j = 1:topo(2)-1
        dist_fence(2*i-1,2*j) = dist(weights(:,i,j),weights(:,i,j+1),dist_threshold);
    end
end 
for i = 1:topo(1)-1 
    for j = 1:topo(2)
        dist_fence(2*i,2*j-1) = dist(weights(:,i,j),weights(:,i+1,j),dist_threshold);
    end
end 
end

function distance = dist(w1,w2,dist_threshold)
distance = 0;
distance = distance + sum((w1-w2).^2);
if distance > dist_threshold
    distance = 10;
else
    distance = 5;
end
end 
%% grid map
function gridmap(grid_mat,iter)
[X,Y]=meshgrid(1:11);
figure; hold on;
plot(X,Y,'k');
plot(Y,X,'k');axis off
Z = zeros(11);
dense = 20;
mid = 5;
for i = 1:10
    for j = 1:10
        if grid_mat(i,j)>=dense
            Z(i,j) = 0.95; 
        elseif and(grid_mat(i,j)<dense, grid_mat(i,j)>=mid)
            Z(i,j) = 0.55;            
        end
    end
end
writeFile(Z);
s = surface(Z);
writeFile(grid_mat);
end
%% Record the data store in a txt
function writeFile(mat)
[m n] = size(mat);
fileID = fopen('/Users/Alan/Desktop/output.txt','a');
str = repmat('%5d ',1,m);
fprintf(fileID,strcat(str,'\n'),mat);
fprintf(fileID,'\n\n\n\n\n');
fclose(fileID);
end 

end