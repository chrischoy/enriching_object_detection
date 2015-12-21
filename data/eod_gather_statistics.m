clear;
addpath('HoG');
addpath('HoG/features');
VOCPath = '/var/tmp/libs/VOC2007/VOCdevkit/VOC2007/JPEGImages/';
v = dir(VOCPath);
file = v(~[v.isdir]);
filenames = {file.name};

padding = 4;
sbin = 4;
nLevel = 5;
param = get_default_params(sbin,nLevel);
param.detect_pyramid_padding = 0;
% Maximum Window Size
N1 = 20;
N2 = 20;
N = max(N1,N2);
d = 31;
N_0 = N1*N2;
mu_0 = zeros(d,1);
numberOfMuData = 0;
visualize = false;
% Make covariance matrix for above and below the current cell
UP = 1;
DOWN = 2;

[~, hostName] = system('hostname');
[s,e] = regexp( hostName, '\d*');
hostName = ['napoli' hostName(s:e)];
save(hostName,'N1');
system(['scp ' hostName '.mat @capri7:/home/chrischoy/Dropbox/Research/ELDA']);
% Since there are N_0 different relative offsets, we can learn Gamma for
% one offset. Gamma(-i,-j) = Gamma(i,j). Assume left right symmetry but not
% up down symmetry
sumHOGcell = zeros(d,1);
compensationSumHOGcell = zeros(d,1); % A running compensation for lost low-order bits.
if ~exist('mu','file')
    for filename = filenames
        fprintf([filename{1} '\n']);
        testDoubleIm = im2double(imread([VOCPath filename{1}]));
        sz = size(testDoubleIm);
        minsz = min(sz(1:2));
        maximumScale = ceil((N+2)*sbin/minsz*10)/10; % borders usually have no features 

        % only search upto 20% of the image size
        param.detect_min_scale = maximumScale;
        [feature_pyramid.hog, feature_pyramid.scales] = esvm_pyramid(testDoubleIm, param);

        % Remove HOG level that has less than N^2 cells (both axis should have
        % at least N cells
        minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), feature_pyramid.hog);
        feature_pyramid.hog = feature_pyramid.hog(minsizes >= N+2);  % borders usually have no features 
        feature_pyramid.scales = feature_pyramid.scales(minsizes >= N+2);  % borders usually have no features 

        for levelIdx = 1:length(feature_pyramid.scales)
    %         imagesc(HOGpicture(features.hog{levelIdx}));
    %         waitforbuttonpress;
            sz = size(feature_pyramid.hog{levelIdx});

            numberOfMuData = numberOfMuData + (sz(1)-2)*(sz(2)-2); % Edges have 0 value so remove them
            % Kahan Summation 
            levelSum = sum(sum(feature_pyramid.hog{levelIdx}(2:end-1,2:end-1,:),1),2);
            y = squeeze(levelSum) - compensationSumHOGcell;
            t = sumHOGcell + y;
            compensationSumHOGcell = (t - sumHOGcell) - y;
            sumHOGcell = t;
            
            if visualize
                subplot(121);
                imagesc(HOGpicture(levelSum/(sz(1)-2)*(sz(2)-2),200));
                title(num2str(levelIdx));
                colorbar;
                pause(0.01);
            end
        end
        
        if visualize
            subplot(122);
            a = zeros(1,1,d);
            a(1,1,:) = sumHOGcell;
            imagesc(HOGpicture(a/numberOfMuData,200));
            title(num2str(numberOfMuData));
            colorbar;
            pause(0.01);
        end
    end
    mu = sumHOGcell / numberOfMuData;
    save('mu.mat','mu','numberOfMuData');
else
    load('mu.mat');
end

% Once we found the the mean, we can compute the covariance matrix
sumGamma{UP} = zeros( d, N1 * N2 * d);
sumGamma{DOWN} = zeros( d, N1 * N2 * d);

compensationGamma{UP} = zeros( d, N1 * N2 * d);
compensationGamma{DOWN} = zeros( d, N1 * N2 * d);

compensationSumHOGcell = zeros(d,1);

numberOfGammaData = 0;

gammaIdx = 1;

scrambleHOGMatrix = zeros(d, d * N1 * N2);
toCovarianceScrambleMatrix = zeros(d * N1, d * N2);
muMatrix = zeros(N1,N2,d);
blockIdx = reshape(1:d^2,d,d);
for i = 1:N1
    for j = 1:N2
        muMatrix(i,j,:) = mu;
        scrambleHOGMatrix(:,((j-1)*N1 + i - 1)*d + 1: ((j-1)*N1 + i - 1)*d + d) = ...
                                        repmat(i + (j-1)*N1 + (0:d-1)*N1*N2, d,1);
        toCovarianceScrambleMatrix((i-1)*d + 1:i*d, (j-1)*d + 1 : j*d) = d^2 * ((i-1) + N1 * (j-1)) + blockIdx;
    end
end

% CreateParpool('/scratch/chrischoy/');
% save([hostName '_pool']);
% system(['scp ' hostName '_pool.mat @capri7:/home/chrischoy/Dropbox/Research/ELDA']);
fileIdx = 0;
for filename = filenames
    fileIdx = 1 + fileIdx;
    tic
    fprintf([filename{1} '\n']);
    testDoubleIm = im2double(imread([VOCPath filename{1}]));
    sz = size(testDoubleIm);
    minsz = min(sz(1:2));
    maximumScale = ceil((N+2)*sbin/minsz*10)/10; % borders usually have no features 
    
    % only search upto 20% of the image size
    param.detect_min_scale = maximumScale;
    [feature_pyramid.hog, feature_pyramid.scales] = esvm_pyramid(testDoubleIm, param);
    
    % Remove HOG level that has less than N^2 cells (both axis should have
    % at least N cells
    minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), feature_pyramid.hog);
    feature_pyramid.hog = feature_pyramid.hog(minsizes >= N+2); % borders usually have no features 
    feature_pyramid.scales = feature_pyramid.scales(minsizes >= N+2);% borders usually have no features 
    
    gammaTempUp = zeros( d, N1 * N2 * d, sz(1)-N1-2);
    gammaTempDown = zeros( d, N1 * N2 * d, sz(1)-N1-2);
    
    for levelIdx = 1:length(feature_pyramid.scales)
        sz = size(feature_pyramid.hog{levelIdx});
        numberOfGammaData = numberOfGammaData + (sz(1)-2-N1)*(sz(2)-2-N2);         
        % parfor i = 2:sz(1)-N1-1
        for i = 2:sz(1)-N1-1
            for j = 2:sz(2)-N2-1
                currWindow = feature_pyramid.hog{levelIdx}(i:i+N1-1,j:j+N2-1,:) - muMatrix;
                leftFeatureUp= squeeze(feature_pyramid.hog{levelIdx}(i,j,:));
                leftFeatureDown = squeeze(feature_pyramid.hog{levelIdx}(i+N1-1,j+N2-1,:));
                mixedWindow = currWindow(scrambleHOGMatrix);
                gammaTempUp(:,:,i-1) = gammaTempUp(:,:,i-1) + bsxfun(@times, (leftFeatureUp-mu), mixedWindow);
                gammaTempDown(:,:,i-1) = gammaTempDown(:,:,i-1) + bsxfun(@times, (leftFeatureDown-mu), mixedWindow);
            end
        end
    end
    
    % Kahan Summation
    for upDown = 1:2
        if upDown == UP
            y = sum(gammaTempUp,3) - compensationGamma{upDown};
        else
            y = sum(gammaTempDown,3) - compensationGamma{upDown};
        end
        t = sumGamma{upDown} + y;
        compensationGamma{upDown} = (t -  sumGamma{upDown}) - y;
        sumGamma{upDown} = t;        
    end
    
    if mod(fileIdx,100) == 1
        gammaFile = sprintf('sumGamma_N1_%d_N2_%d_sbin_%d_nLevel_%d_nImg_%d_%s.mat',N1,N2,sbin,nLevel,fileIdx,hostName);
        save(gammaFile, 'sumGamma','N1','N2','d','numberOfGammaData','sbin','nLevel');
        system(['scp ' gammaFile ' @capri7:/home/chrischoy/Dropbox/Research/ELDA']);
    end
    
    if visualize
        subplot(121);
        imagesc(abs(sumGamma{UP}(toCovarianceScrambleMatrix))/numberOfGammaData);
        axis equal; axis tight; colorbar;
        title(num2str(numberOfGammaData));

        subplot(122);
        imagesc(abs(sumGamma{DOWN}(toCovarianceScrambleMatrix))/numberOfGammaData);
        axis equal; axis tight; colorbar;
        title(num2str(numberOfGammaData));
        pause(0.1);
    end
    toc;
end

gammaFile = sprintf('sumGamma_N1_%d_N2_%d_sbin_%d_nLevel_%d_%s.mat',N1,N2,sbin,nLevel,hostName);
save(gammaFile, 'sumGamma','N1','N2','d','numberOfGammaData','sbin','nLevel');
system(['scp ' gammaFile ' @capri7:/home/chrischoy/Dropbox/Research/ELDA']);