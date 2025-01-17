
python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
export PS_VERBOSE=1
../../tools/launch.py -n 12 -s 12 -i eth1 --launcher ssh -H hosts \
 python train_cifar10.py --data-dir /home/mxnet_data/cifar10/ \
 --batch-size 45 --lr 0.1 --lr-factor .94 --num-epoch 1 --num-examples 50000 \
 --gpu 0,1 --kv-store dist_async \