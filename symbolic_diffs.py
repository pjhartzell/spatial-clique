import math
from sympy import symbols, diff, sqrt

xsrc, ysrc, zsrc, xdst, ydst, zdst = symbols('xsrc ysrc zsrc xdst ydst zdst', real=True)

# distance2d = sqrt( (xdst-xsrc)**2 + (ydst-ysrc)**2 )
# print("diff xsrc = {}".format(diff(distance2d, xsrc)))
# print("diff ysrc = {}".format(diff(distance2d, ysrc)))
# print("diff xdst = {}".format(diff(distance2d, xdst)))
# print("diff ydst = {}".format(diff(distance2d, ydst)))

distance3d = sqrt( (xdst-xsrc)**2 + (ydst-ysrc)**2 + (zdst-zsrc)**2 )
print("diff xsrc = {}".format(diff(distance3d, xsrc)))
print("diff ysrc = {}".format(diff(distance3d, ysrc)))
print("diff zsrc = {}".format(diff(distance3d, zsrc)))
print("diff xdst = {}".format(diff(distance3d, xdst)))
print("diff ydst = {}".format(diff(distance3d, ydst)))
print("diff zdst = {}".format(diff(distance3d, zdst)))