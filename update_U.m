function U = update_U(X, Z, w, h, K, N, S, O)
    D = zeros(N, K);
    for s = 1:S
        for k = 1:K
            diff2 = (X{s} - Z{s}(k,:)).^2;        % N x O_s
            weighted_diff = diff2 * w{s}(k,:)';   % N x 1
            dist_sum = sum(weighted_diff, 2);     % N x 1
            D(:,k) = D(:,k) + h(k,s) .* dist_sum; % sum over views
        end
    end
    [~, idx] = min(D, [], 2);
    U = full(sparse(1:N, idx, 1, N, K));  % hard assignment
end