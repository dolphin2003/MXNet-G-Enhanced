
# pylint: disable=too-many-instance-attributes, too-many-arguments
"""A `BucketingModule` implement the `BaseModule` API, and allows multiple
symbols to be used depending on the `bucket_key` provided by each different
mini-batch of data.
"""

import logging

from .. import context as ctx

from ..initializer import Uniform

from .base_module import BaseModule
from .module import Module

class BucketingModule(BaseModule):
    """A bucketing module is a module that support bucketing.

    Parameters