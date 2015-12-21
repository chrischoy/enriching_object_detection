function [detectors, detector_table]= dwot_make_table_from_detectors(detectors, detector_table)

if nargin == 1
  detector_table = containers.Map;
end

n_detectors = numel(detectors);
n_table = detector_table.Count;

for i = (n_table + 1):n_detectors
  detector_key = dwot_detector_key( detectors{i}.az, detectors{i}.el, detectors{i}.yaw, detectors{i}.fov, [1] );
  detector_table(detector_key) = i;
end

