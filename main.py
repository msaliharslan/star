#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:19:58 2020

@author: salih
"""

import sys
sys.path.append("Evaluators")

import fetcher
import tester

fetcher.fetchAllData()
tester.plotReprojectionErrorBoth()
# tester.plot_repro_err_acc.savefig("Figure1.png") maybe we can also save the figures in recorder


