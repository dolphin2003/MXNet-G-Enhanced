python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
python ./scp_groups.py
export PS_VERBOSE=1
../../tools/launch.py -n 12 -s 12 -i eth1 --launcher ssh -H hosts \
 