# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Python functions for directly manipulating TFRecord-formatted files.
"""

from __future__ import print_function as _print_function

from tensorflow.python.lib.io.python_io import TFRecordCompressionType
from tensorflow.python.lib.io.python_io import TFRecordOptions
from tensorflow.python.lib.io.python_io import TFRecordWriter
from tensorflow.python.lib.io.python_io import tf_record_iterator

del _print_function

import sys as _sys
from tensorflow.python.util import deprecation_wrapper as _deprecation_wrapper

if not isinstance(_sys.modules[__name__], _deprecation_wrapper.DeprecationWrapper):
  _sys.modules[__name__] = _deprecation_wrapper.DeprecationWrapper(
      _sys.modules[__name__], "python_io")
