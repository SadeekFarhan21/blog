"""Keep generated Python bytecode out of the source tree during tests."""

import sys


sys.dont_write_bytecode = True
