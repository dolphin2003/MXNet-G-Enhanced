# Char LSTM Example.

# This example aims to show how to use lstm to build a char level language model, and generate text from it. We use a tiny shakespeare text for demo purpose.
# Data can be found at https://github.com/dmlc/web-data/tree/master/mxnet/tinyshakespeare. 

# If running for the first time, download the data by running the following commands: sh get_ptb_data.sh
 
require(mxnet)
source("lstm.R")

# Set basic network parameters.
batch.size = 32
seq.len = 32
num.hidden = 256
num.embed = 256
num.lstm.layer = 2
num.round = 21
learning.rate= 0.01
wd=0.00001
clip_gradient=1
update.period = 1

# Make dictionary from text
make.dict <- function(text, max.vocab=10000) {
	text <- strsplit(text, '')
	dic <- list()
	idx <- 1
    for (c in text[[1]]) {
    	if (!(c %in% names(dic))) 