clc; clear; close all;

%% setting input singal

freq = [0.1; 0.22; 0.37]*2*pi;
ampAbs = [1; 1; 1];
amp = ampAbs.*exp(1j*2*pi*rand(length(ampAbs), 1));
N = 32; n = (0: N-1)';
snrDb = 10; 
y = exp(1j*n*freq')*amp;
yNoise = awgn(y, snrDb, 'measured');

%% FFT initialization
numExpWavHat = 3;
fftLen = 4*N;
[ampFft, freqFft, yFft] = FFT_initializer(yNoise, numExpWavHat, fftLen);

%% use MNN to estimate line spectral
tol = zeros(4,1);
tol(1) = 1e-6;
tol(2) = N*1e-6; % set stopping criterion
tol(3) = 1e-6; % node merge tol
tol(4) = 1e-6; % node prune tol
learningRate = 2e-4; % learning rate
momentumRate = 0.99; % momentum rate

[ampEst, freqEst, lossRcd] = MNN_estimator(yNoise, ampFft, freqFft, tol, ...
    learningRate, momentumRate);

%% plot result
H = figure(); 
semilogy(0:1/fftLen:1-1/fftLen, abs(yFft),'-','LineWidth',1.5);
ylim([1e-4,100]); hold on;
stem(freq/(2*pi), abs(amp), 'r*','LineWidth', 2); 
stem(freqEst/(2*pi), abs(ampEst),'bs','MarkerSize',14,'LineWidth', 1.5);
legend('FFT', 'Ground Truth', 'MNN-based Method');
xlabel('Normalized Frequency'); ylabel('Amplitude');