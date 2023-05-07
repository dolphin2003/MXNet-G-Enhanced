import os
import threading
import getpass
from sets import Set
import socket
import struct
import fcntl

def scp_groups(ip, path, user):
    from_path = './data_sharding'
    to_path = user + '@' + ip + ':' + path
    cmd = 'scp' + ' ' + from_path + ' ' + to_path
    os.system(cmd)

if __name__ == '__main__':

    nodes = []
    with open('./hosts', 'r') as file:
        nodes = [line.strip() for line in file]
    nodes = list(Set(nodes))

    interface = 'ib0'
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    inet = fcntl.ioctl(s.fileno(), 0x8915, struct.pack