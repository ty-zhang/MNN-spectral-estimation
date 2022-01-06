function [ampEst, freqEst, lossRcd] = MNN_estimator(sig, ampInit, freqInit, tol, ...
    learningRate, momentumRate)
    
    ismp = 1; % merge and prune flag
    isplot = 1; % plot loss curve flag

    lossRcd = [];
    N = length(sig); n = (0: N-1)';

    if nargin < 5
        learningRate = 1e-4;
        momentumRate = 0.99;
    end

    while ismp == 1
        [ampEst, freqEst, lossOneStep] = MNN_estimator_onestep(sig, ampInit, freqInit, tol(1:2), ...
            learningRate, momentumRate);
        lossRcd = [lossRcd, lossOneStep];

        % merge node
        [ampEstMp, freqEstMp, mergeFlag] = node_merge(sig, ampEst, freqEst, tol(3));

        % prune node
        pruneTol = N/(N - max(mergeFlag))*finv(1 - tol(4), 2, 2*(N - max(mergeFlag)));
        fftMat = exp(1j*n*freqEstMp');
        sigEst = fftMat*ampEstMp;
        res2 = norm(sig - sigEst, 2)^2;
        spectral = fftMat'*sig;
        pruneFlag = find(abs(spectral).^2/res2 > pruneTol);

        if isempty(pruneFlag) == 0
            ampEstMp = ampEstMp(pruneFlag);
            freqEstMp = freqEstMp(pruneFlag);
        end

        ismp = (length(ampEstMp) < length(ampEst));
        ampInit = ampEstMp;
        freqInit = freqEstMp;
    end

    if isplot
        figure(); semilogy(lossRcd,'LineWidth',1.5); grid on;
    end
end

function [ampEst, freqEst, lossRcd] = MNN_estimator_onestep(sig, ampInit, freqInit, tol, ...
            learningRate, momentumRate)
    isplot = 0; % plot loss curve flag
    N = length(sig); n = (0: N-1)';

    if nargin < 5
        learningRate = 1e-4;
        momentumRate = 0.99;
    end

    maxIterNum = 1e8;
    dAmp = 0; dFreq = 0;
    ampEst = ampInit; freqEst = freqInit;
    lossOld = inf; lossRcd = zeros(1, maxIterNum);

    for ii = 1:maxIterNum
        fftMat = exp(1j*n*freqEst');
        sigEst = fftMat*ampEst;
        err = sigEst - sig;
        partialAmp = fftMat'*err;
        partialFreq = -2*imag(ampEst.*(fftMat.'* (n.*conj(err))));

        dAmp = momentumRate*dAmp + (1 - momentumRate)*partialAmp;
        dFreq = momentumRate*dFreq + (1 - momentumRate)*partialFreq;

        ampEst = ampEst - learningRate*dAmp;
        freqEst = freqEst - learningRate*dFreq;

        loss = norm(err, 2)^2;
        lossRcd(ii) = loss;

        % stopping criterion
        if norm(dFreq) < tol(1) && norm(dAmp) < tol(1)
            break;
        end
        if mod(ii, 1e2) == 0 || ii == 1
            if abs(lossOld - loss) < tol(2)
                break;
            end
            lossOld = loss;
        end
    end

    lossRcd = lossRcd(1: ii);

    if isplot
        figure(); semilogy(lossRcd,'LineWidth',1.5); grid on;
    end
end

function [ampEstM, freqEstM, flag] = node_merge(sig, ampEst, freqEst, errRate)
    N = length(sig); n = (0: N-1)';
    [freqEst, order] = sort(freqEst, 'ascend');
    ampEst = ampEst(order);
    numExpWav = length(ampEst);
    flag = zeros(1, numExpWav);
    flag(1) = 1;

    sigEst = exp(1j*n*freqEst')*ampEst;
    noise = sig - sigEst;
    sigma2 = 1/N*norm(noise, 2)^2;

    rho1 = sum(n.^2);
    gaussianErr = norminv(errRate);
    
    for mm = 2:numExpWav
        dFreq = freqEst(mm) - freqEst(mm-1);
        amp1 = ampEst(mm-1); amp2 = ampEst(mm);

        rho2 = sum(exp(1j*dFreq*n).*(n.^2));

        numeCRB = (abs(amp1)^2 + abs(amp2))*rho1 + 2*real(amp1'*amp2*rho2);
        denoCRB = abs(amp1)^2*abs(amp2)^2*rho1^2 - real(amp1'*amp2*rho2)^2;
        CRB = (sigma2/2)*numeCRB/denoCRB;

        mergeCriterion = -sqrt(CRB)*gaussianErr;

        if dFreq > mergeCriterion
            % do not merge
            flag(mm) = flag(mm-1) + 1;
        else
            % merge
            flag(mm) = flag(mm-1);
        end
    end

    ampEstM = zeros(max(flag), 1);
    freqEstM = zeros(max(flag), 1);

    for mm = 1:max(flag)
        ampEstM(mm) = sum(ampEst(find(flag == mm)));
        freqEstM(mm) = mean(freqEst(find(flag == mm)));
    end
end