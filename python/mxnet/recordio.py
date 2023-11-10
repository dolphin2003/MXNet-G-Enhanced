# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments, no-member

"""Python interface for DLMC RecrodIO data format"""
from __future__ import absolute_import
from collections import namedtuple

import os
import ctypes
import struct
import numbers
import numpy as np

from .base import _LIB
from .base import RecordIOHandle
from .base import check_call
from .base import c_str
try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False

class MXRecordIO(object):
    """Python interface for read/write RecordIO data formmat

    Parameters
    ----------
    uri : string
        uri path to recordIO file.
    flag : string
        "r" for reading or "w" writing.
    """
    def __init__(self, uri, flag):
        self.uri = c_str(uri)
        self.handle = RecordIOHandle()
        self.flag = flag
        self.is_open = False
        self.open()

    def open(self):
        """Open record file"""
        if self.flag == "w":
            check_call(_LIB.MXRecordIOWriterCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = True
        elif self.flag == "r":
            check_call(_LIB.MXRecordIOReaderCreate(self.uri, ctypes.byref(self.handle)))
            self.writable = False
        else:
            raise ValueError("Invalid flag %s"%self.flag)
        self.is_open = True

    def __del__(self):
        self.close()

    def close(self):
        """close record file"""
        if not self.is_open:
            return
        if self.writable:
            check_call(_LIB.MXRecordIOWriterFree(self.handle))
        else:
            check_call(_LIB.MXRecordIOReaderFree(self.handle))

    def reset(self):
        """Reset pointer to first item. If record is opened with 'w',
        this will truncate the file to empty"""
        self.close()
        self.open()

    def write(self, buf):
        """Write a string buffer as a record

        Parameters
        ----------
        buf : string (python2), bytes (python3)
            buffer to write.
        """
        assert self.writable
        check_call(_LIB.MXRecordIOWriterWriteRecord(self.handle,
                                                    ctypes.c_char_p(buf),
                                                    ctypes.c_size_t(len(buf))))

    def read(self):
        """Read a record as string

        Returns
        ----------
        buf : string
            buffer read.
        """
        assert not self.writable
        buf = ctypes.c_char_p()
        size = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOReaderReadRecord(self.handle,
                                                   ctypes.byref(buf),
                                                   ctypes.byref(size)))
  