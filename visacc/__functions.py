# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 11:30:52 2017

@author: rook: https://stackoverflow.com/users/2242445/rook
    The code
@author: Leevi Annala
    Comments
"""

def line(p1, p2):
    '''
    :param p1: (x, y)-coordinates of a point for a line
    :param p2: (x, y)-coordinates of a point for a line
    :return: parametrisation of a line that goes throug p1 and p2
    Copied directly from stackoverflow: https://stackoverflow.com/a/20679579
    '''
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    '''
    :param L1: a line
    :param L2: another line
    :return: their crossing point if it exists and is just one, False otherwise
    Copied directly from stackoverflow: https://stackoverflow.com/a/20679579
    '''
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False
