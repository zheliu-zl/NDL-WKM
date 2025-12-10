function J = calculate_J(X, Z, U, w, h, K, S, O)
    J = 0;
    for s = 1:S
        for k = 1:K
            diff2 = (X{s} - Z{s}(k,:)).^2;
            dist_sum = sum((diff2 * w{s}(k,:)'), 2);
            J = J + sum(U(:,k) .* h(k,s) .* dist_sum);
        end
    end
end