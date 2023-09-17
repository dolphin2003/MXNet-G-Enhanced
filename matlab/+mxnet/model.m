classdef model < handle
%MODEL MXNet model, supports load and forward

properties
% The symbol definition, in json format
  symbol
% parameter weights
  params
% whether or not print info
  verbose
end

properties (Access = private)
% mxnet predictor
  predictor
% the previous input size
  prev_input_size
% the previous device id
  prev_dev_id
% the previous device type (cpu or gpu)
  prev_dev_type
% the previous output layers
  prev_out_layers
end

methods
  function obj = model()
  %CONSTRUCTOR
  obj.predictor = libpointer('voidPtr', 0);
  obj.prev_input_size = zeros(1,4);
  obj.verbose = 1;
  obj.prev_dev_id = -1;
  obj.prev_dev_type = -1;
  end

  function delete(obj)
  %DESTRUCTOR
  obj.free_predictor();
  end

  function load(obj, model_prefix, num_epoch)
  %LOAD load model from files
  %
  % A mxnet model is 