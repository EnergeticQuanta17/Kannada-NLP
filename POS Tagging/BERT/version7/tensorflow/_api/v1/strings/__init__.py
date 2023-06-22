# This file is MACHINE GENERATED! Do not edit.
# Generated by: tensorflow/python/tools/api/generator/create_python_api.py script.
"""Operations for working with string Tensors.
"""

from __future__ import print_function as _print_function

from tensorflow.python import as_string
from tensorflow.python import reduce_join
from tensorflow.python import regex_full_match
from tensorflow.python import regex_replace
from tensorflow.python import string_format as format
from tensorflow.python import string_join as join
from tensorflow.python import string_length as length
from tensorflow.python import string_lower as lower
from tensorflow.python import string_strip as strip
from tensorflow.python import string_to_hash_bucket_fast as to_hash_bucket_fast
from tensorflow.python import string_to_hash_bucket_strong as to_hash_bucket_strong
from tensorflow.python import string_to_hash_bucket_v1 as to_hash_bucket
from tensorflow.python import string_to_number_v1 as to_number
from tensorflow.python import string_upper as upper
from tensorflow.python import substr
from tensorflow.python import unicode_script
from tensorflow.python import unicode_transcode
from tensorflow.python.ops.ragged.ragged_string_ops import string_bytes_split as bytes_split
from tensorflow.python.ops.ragged.ragged_string_ops import strings_split_v1 as split
from tensorflow.python.ops.ragged.ragged_string_ops import unicode_decode
from tensorflow.python.ops.ragged.ragged_string_ops import unicode_decode_with_offsets
from tensorflow.python.ops.ragged.ragged_string_ops import unicode_encode
from tensorflow.python.ops.ragged.ragged_string_ops import unicode_split
from tensorflow.python.ops.ragged.ragged_string_ops import unicode_split_with_offsets

del _print_function

import sys as _sys
from tensorflow.python.util import deprecation_wrapper as _deprecation_wrapper

if not isinstance(_sys.modules[__name__], _deprecation_wrapper.DeprecationWrapper):
  _sys.modules[__name__] = _deprecation_wrapper.DeprecationWrapper(
      _sys.modules[__name__], "strings")
