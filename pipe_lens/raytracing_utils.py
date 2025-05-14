import numpy as np

from numpy import pi, sin, sqrt

def roots_bhaskara(a, b, c):
    '''Computes the roots of the polynomial ax^2 + bx + c = 0'''

    sqdelta = sqrt(b ** 2 - 4 * a * c)
    x1 = (-b + sqdelta) / (2 * a)
    x2 = (-b - sqdelta) / (2 * a)
    return x1, x2


def rhp(x):
    '''Projects an angle to the Right Half Plane [-pi/2; pi/2]'''
    x = np.mod(x, pi)
    x = x - (x > pi / 2) * pi
    x = x + (x < -pi / 2) * pi
    return x


def uhp(x):
    '''Projects an angle to the Upper Half Plane [0; pi]'''
    x = rhp(x)
    x = x + (x < 0) * pi
    return x


def snell(v1, v2, gamma1, dydx):
    """Computes the the new angle  after the refraction
  of the first angle with the Snell's law (top-down)"""
    gamma1 = uhp(gamma1)
    slope = rhp(np.arctan(dydx))
    normal = slope + pi / 2
    theta1 = gamma1 - normal
    arg = sin(theta1) * v2 / v1
    bad_index = np.abs(arg) > 1
    # Forcing the argument to be always within the [-1, 1] interval:
    arg[bad_index] = np.tanh(arg[bad_index])
    theta2 = np.arcsin(arg)
    gamma2 = slope - pi / 2 + theta2
    return gamma2, theta1, theta2



