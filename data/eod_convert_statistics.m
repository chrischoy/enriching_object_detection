% Convert collected statistics

stat1601 = load('sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_1601_napoli1_gamma.mat');
stat3601 = load('sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_3601_napoli3_from_1601.mat');

gamma_concat = stat3601.sumGamma;
N1 = stat3601.N1;
N2 = stat3601.N2;
d = stat3601.d;
gamma = zeros(N1 * d, N2 * d);


for i = 1:N1
    for j = 1:N2
        gamma( (i-1)*d + 1 : i*d, (j-1)*d + 1:j*d) =...
            gamma_concat(:, ((j-1)*N1 + i - 1)*d + 1: ((j-1)*N1 + i - 1)*d + d);
        % ((i-1)*N1 + j-1)*d + 1: ((i-1)*N1 + j)*d);
    end
end

gamma_concat = stat1601.sumGamma{1};

for i = 1:N1
    for j = 1:N2
        gamma( (i-1)*d + 1 : i*d, (j-1)*d + 1:j*d) =...
            gamma( (i-1)*d + 1 : i*d, (j-1)*d + 1:j*d) + ...
            gamma_concat(:, ((j-1)*N1 + i - 1)*d + 1: ((j-1)*N1 + i - 1)*d + d);
        % ((i-1)*N1 + j-1)*d + 1: ((i-1)*N1 + j)*d);
    end
end

imagesc(gamma); axis equal; axis tight;
Gamma = gamma / (stat3601.numberOfGammaData + stat1601.numberOfGammaData);
mu = stat1601.mu;

save('sumGamma_N1_40_N2_40_sbin_4_nLevel_10_nImg_3601_napoli3_gamma.mat','Gamma','mu')