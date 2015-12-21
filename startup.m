% Setup search paths
if ~exist('path_setup', 'var')
  global path_setup PATH_TO_RENDERER
  PATH_TO_RENDERER = '../OSGRenderer/';

  addpath('bin');
  addpath('lib');
  addpath('lib/HoG');
  addpath('util');
  % Add binary path to the current search path
  addpath([PATH_TO_RENDERER]);
  addpath([PATH_TO_RENDERER, 'bin/']);
end

