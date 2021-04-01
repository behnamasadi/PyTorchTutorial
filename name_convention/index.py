# PEP 8: https://www.python.org/dev/peps/pep-0008/

# Correct:
import os
import sys

# Wrong:
import sys, os

# classes:
# should normally use the CapWords convention., However, it still seems that builtin or extension types written
# in C are more likely to have lowercase names (e.g., numpy.array, not numpy.Array).

# Package and Module Names:
# Modules should have short, all-lowercase names.

# Type Variable Names:
# Names of type variables introduced use CapWords

# Function and Variable Names:
# Function names should be lowercase, with words separated by underscores as necessary to improve readability.

# Constants:
# Constants are usually defined on a module level and written in all capital letters with underscores
# separating words. Examples include MAX_OVERFLOW and TOTAL.

