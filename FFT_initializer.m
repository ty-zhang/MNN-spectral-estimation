function [ampFft, freqFft, yFft] = FFT_initializer(y, numExpWav, fftLen)
    %% use FFT to initialize the MNN
    N = length(y); n = (0:N-1)';
    yFft = 1/N*fft(y, fftLen);
    [~, freqFft] = findpeaks(abs(yFft), 'SortStr','descend');
    freqFft = freqFft(1: numExpWav);

    %% split the FFT spectrum peaks
    freqFft2 = [];
    for mm = 1:numExpWav
        freqTmp = (freqFft(mm)+1).*(abs(yFft((freqFft(mm)+1)))>=abs(yFft((freqFft(mm)-1)))) + ...
            (freqFft(mm)-1).*(abs(yFft((freqFft(mm)+1)))<abs(yFft((freqFft(mm)-1))));
        freqFft2 = [freqFft2; freqFft(mm); freqTmp];
    end
    freqFft = freqFft2;
    freqFft = unique(freqFft, 'stable');
    freqFft = 2*pi*(freqFft-1)/fftLen;
    ampFft = zeros(length(freqFft), 1);

    for mm = 1:numExpWav
        fftMat = exp(1j*freqFft(mm)*n);
        ampFft(mm) = fftMat\y;
    end
end