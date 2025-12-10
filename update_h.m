function h = update_h(X, Z, U, w, K, S, O)
    Dks = zeros(K, S);
    for s = 1:S
        for k = 1:K
            diff2 = (X{s} - Z{s}(k,:)).^2;
            Dks(k,s) = sum(U(:,k) .* (diff2 * w{s}(k,:)'));
        end
    end
    Dks(Dks == 0) = eps;
    
    h = zeros(K, S);
    for k = 1:K
        geomMean = prod(Dks(k,:))^(1/S);
        for s = 1:S
            h(k,s) = geomMean / Dks(k,s);
        end
    end
end