#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:59:55 2023

@author: martina
"""

from collections import namedtuple

Param = namedtuple('Param', ('EI', 'L', 'n_dim', 'n_nodes', 'g', 'k', 'p', 'r1', 'r2'))
param_all = Param(EI=1000, L=3, n_dim = 2, n_nodes=31, g=-9.81, k=1000, p=1000000, r1=2, r2=-2)
