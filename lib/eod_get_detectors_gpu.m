function templates_gpu = eod_get_detectors_gpu(detectors)
%EOD_GET_DETECTORS_GPU move detectors to GPU

templates_gpu = cellfun(@(x) gpuArray(single(x.whow(end:-1:1,end:-1:1,:))), detectors,'UniformOutput',false);
