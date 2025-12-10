function Z = update_Z(X, U, K, S, O)
    Z = cell(1, S);
    for s = 1:S
        Z{s} = zeros(K, O(s));
        for k = 1:K
            num = U(:,k)' * X{s};     % 1 x O_s
            den = sum(U(:,k));        % scalar
            Z{s}(k,:) = num ./ den;   % 1 x O_s
        end
    end
end