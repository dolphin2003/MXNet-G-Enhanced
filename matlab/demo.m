%% Download sample image and model
if ~exist('cat.png', 'file')
  assert(~system('wget --no-check-certificate https://raw.githubusercontent.com/dmlc/mxnet.js/master/data/cat.png'));
end

if ~exist('model/Inception_BN-0039.params', 'file')
  assert(~system('wget --no-check-certificate https://s3.amazonaws.com/dmlc/model/inception-bn.tar.gz'));
  assert(~system('tar -zxvf inception-bn.tar.gz'))
end

%% Load the model
clear model
model = mxnet.model;
model.load('model/Inception_BN', 39);

%% Load and resize the image
img = imresize(imread('cat.png'), [224 224]);
img = single(img) - 120;
%% Run prediction
pred = model.forward(img);

%% load the labels
labels = {};
fid = fopen('model/synset.txt', 'r');
assert(fid >= 0);
tline = fgetl(fid);
while ischar(tline)
  labels{end+1} = tline;
  tline = fgetl(fid);
end
fclose(fid);

%% find the predict label
[p, i] = max(pred);
fprintf('the