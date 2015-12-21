function [boxes, scores, azs, els, yaws, fovs, renderings, depths] = eod_extract_proposals(proposals)

boxes = cell2mat(cellfun(@(x) x.image_bbox, proposals','uniformoutput',false));
scores = cellfun(@(x) x.score, proposals');

azs = cellfun(@(x) x.x(1), proposals');
els = cellfun(@(x) x.x(2), proposals');
yaws = cellfun(@(x) x.x(3), proposals');
fovs = cellfun(@(x) x.x(4), proposals');

renderings = cellfun(@(x) im2double(x.rendering_image), proposals','uniformoutput',false);
depths = cellfun(@(x) x.rendering_depth, proposals','uniformoutput',false);
