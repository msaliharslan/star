#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:19:58 2020

@author: salih
"""

import sys
sys.path.append("Evaluators")

import argparse
import fetcher
import tester

parser = argparse.ArgumentParser()
parser.add_argument( "--focusDir", nargs = 1 )
args = parser.parse_args()

fetcher.insertFocusDataPath(args.focusDir[0])
fetcher.fetchAllData()
tester.plotReprojectionErrorBoth()
# tester.plot_repro_err_acc.show()
# tester.plot_repro_err_gyro.show()
# tester.plot_repro_err_acc.savefig("Figure1.png") #maybe we can also save the figures in recorder

