"""remapper.py
YOU DO NOT NEED TO EDIT OR TURN IN THIS FILE.

Provides a function 'map' that will take a point
in the x-y plane and return two new coordinates, basically
corresponding to its polar coordinates theta and rho.
"""

import math


def remap(x, y):
    angle = math.atan2(x, y)
    radius = math.sqrt(x * x + y * y)
    return [angle, radius]
