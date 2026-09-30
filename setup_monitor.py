# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:44:06 2026

@author: becev
"""

from psychopy import monitors

mon = monitors.Monitor('taVNS_lab')
mon.setWidth(53.0)            # visible screen width, cm
mon.setDistance(60.0)         # eye to screen, cm
mon.setSizePix([1920, 1080])
mon.save()