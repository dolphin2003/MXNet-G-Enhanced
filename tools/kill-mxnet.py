#!/usr/bin/env python

import os, sys
import threading
from sets import Set

if len(sys.argv) != 2:
  print "usage: %s <hostfile>" % sys.argv[0]
  sys.exit(1)

host_file = sys.argv[1]
prog_name = "train"

# Get host IPs
with open(host_file, "r") as f:
  hosts = f.read().splitlines()
hosts = list(Set(hosts))
ssh_cmd = (
    "ssh "
    "-o StrictHostKeyChecking=no "
    "-o