#2017.05  zhujian
#filter out the speed of all nodes from the log file,
#compute the average speed,then write it in .txt file
import numpy as np
import re
import getpass

def filter_out_Node(lineInfo):
	w1="Node"
	w2=" Epoch"
	start = lineInfo.find(w1)
	start += len(w1)
	end = lineInfo.find(w2)
	return lineInfo[start:end].strip('[]')

def filter_out_speed(lineInfo):
	w1="Speed: "
	w2="samples"
	start = lineInfo.find(w1)
	if start