clear all; clc;
 Data = load("datasets\foresttype.txt");  Y = Data(:,end);
 Data = mapminmax(Data',0,1); Data = Data';Data(:,end) = Y;
 view = 2; features = [9 18];
 left = 1;right=0; X=cell(1,view);
 for i=1:view
     X{i} = Data(:, left:features(i)+right);
     left = features(i)+right+1;
     right = features(i)+right;
 end

% load datasets\MSRC-v1.mat
% Y = Y + 1;
% data = X;
% view = 3;
% features = [24,256,254];
% X = {};
% X1 = mapminmax(data{1},0,1);
% X{1} = X1;
% X1 = mapminmax(data{3},0,1);
% X{2} = X1;
% X1 = mapminmax(data{4},0,1);
% X{3} = X1;

% load datasets\SensIT.mat
% Y = label;data = cell(2);
% data{1} = Acou;data{2} = Seis;
% view = 2; features = [50,50];
% for i =1:view
%     X1 = mapminmax(data{i}',0,1);
%     X{i} = X1';
% end

metrics = [];
%% initialization
[n, ~] = size(X{1}); 
K = max(Y); 
S = view;
O = cellfun(@(x) size(x, 2), X);
N = size(X{1}, 1);
U = ones(N, K);
U = U ./ sum(U, 2);
Z = cell(1, S);
for s = 1:S
    [~,Z{s}] = kmeans(X{s},K);
end
h = ones(K, S);
w = cell(1, S);
for s = 1:S
    w{s} = ones(K,O(s)); 
end
J_value = [];
loop = 0;
%% Start iteration
while true
    loop = loop + 1;
    U = update_U(X, Z, w, h, K, N, S, O);
    Z = update_Z(X, U, K, S, O);
    h = update_h(X, Z, U, w, K, S, O);
    w = update_w(X, Z, U, h, K, S, O);
    new_target = calculate_J(X, Z, U, w, h, K, S, O);
    J_value = [J_value, new_target];
    if loop > 1 && abs(J_value(end) - J_value(end - 1)) < 1e-5
        break
    end
end
%% Calculate performance metrics
label = zeros(n, 1);
for i = 1:n
    [~, p] = max(U(i, :));
    label(i) = p;
end
result_label = label_map(label, Y);
result = CalcMeasures(Y, result_label);
[P, R1, F, RI, FM, new_target] = Evaluate(Y, result_label);
fprintf("ACC:%f, NMI:%f, P:%f, R:%f, F:%f, RI:%f, FM:%f, J:%f\n", ...
    result(1), result(2), P, R1, F, RI, FM, new_target);
metrics = [metrics; [result(1), result(2), P, R1, F, RI, FM, new_target]];