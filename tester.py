#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:08:34 2020

@author: salih
"""

import rotationMatrix
import utility
import numpy as np
import matplotlib.pyplot as plt

R_acc = None
R_acc_orthogonalized = None
R_gyro = None
R_gyro_orthogonalized = None
plot_repro_err_acc = None
plot_repro_err_gyro = None

def figureCounter():
    i = 0
    while True:
        yield i
        i += 1

figCounter = figureCounter()
    
def isRotationMatrix(R) :
    """

    Parameters
    ----------
    R : 3 x 3 matrix

    Returns
    -------
    True if the input is a rotation matrix

    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def assignRotationMatrixAcc():
    """
    Assigns the rotation matrix calculated from acc data to R_acc

    Returns
    -------
    None.

    """
    global R_acc
    R_acc = rotationMatrix.calculateRotationMatrixFromAccs(120, 100)
    
def assignRotationMatrixGyro():
    """
    Assigns the rotation matrix calculated from gyro data to R_gyro
    
    Returns
    -------
    None.

    """
    global R_gyro
    R_gyro = rotationMatrix.calculateRotationMatrixFromGyros(400, 300)

def assignRotationMatrixeBoth():
    """
    Assign calculation of rotation matrixes to variables


    Returns
    -------
    None.

    """
    assignRotationMatrixAcc()
    assignRotationMatrixGyro()
    
def orthogonalizeRotationMatrixAcc():
    """
    Chekcs whether the calculated rotation matrix for acc is orthogonal or not.
    If not tries to make it orthogonal. Prints the results.

    Returns
    -------
    None.

    """
    
    global R_acc
    global R_acc_orthogonalized
    
    if (type(R_acc) == type(None) or type(R_gyro) == type(None)):
        assignRotationMatrixAcc()
    
    if ( isRotationMatrix(R_acc) ):
        print( "Calculated rotation matrix from acc data is orthogonal." )
        R_acc_othogonalized = R_acc
    else:
        R_acc_orthogonalized = utility.orthogonalize(R_acc)
        if (isRotationMatrix(R_acc_orthogonalized) ):
            print( "Calculated rotation matrix from acc data is orthogonalized." )
        else:
            print( "Calculated rotation matrix from acc data cannot be orthogonalized!!" )
            
def orthogonalizeRotationMatrixGyro():
    """
    Chekcs whether the calculated rotation matrix for gyro is orthogonal or not.
    If not tries to make it orthogonal. Prints the results.

    Returns
    -------
    None.

    """
    
    global R_gyro
    global R_gyro_orthogonalized
    
    if (type(R_gyro) == type(None) or type(R_gyro) == type(None)):
        assignRotationMatrixGyro()
    
    if ( isRotationMatrix(R_gyro) ):
        print( "Calculated rotation matrix from gyro data is orthogonal." )
        R_gyro_orthogonalized = R_gyro
    else:
        R_gyro_orthogonalized = utility.orthogonalize(R_gyro)
        if (isRotationMatrix(R_gyro_orthogonalized) ):
            print( "Calculated rotation matrix from gyro data is orthogonalized." )
        else:
            print( "Calculated rotation matrix from gyro data cannot be orthogonalized!!" )


def orthogonalizeRotationMatrixBoth():
    """
    Chekcs whether the calculated rotation matrixes are orthogonal or not.
    If not tries to make them orthogonal. Prints the results.

    Returns
    -------
    None.

    """
    orthogonalizeRotationMatrixAcc()
    orthogonalizeRotationMatrixGyro()

def plotReprojectionErrorAcc():
    """
    Calculates and plots the reprojection error using orthogonalized rotation 
    matrix from acc data

    Returns
    -------
    None.

    """
    global R_acc_orthogonalized, plot_repro_err_acc, figCounter
    
    if (type(R_acc_orthogonalized) == type(None)):
        orthogonalizeRotationMatrixAcc()

    repro_err_mag_acc = []
    repro_err_ang_acc = []
    repro_err_x_acc = []
    repro_err_y_acc = []
    repro_err_z_acc = []
    
    for v1,v2 in rotationMatrix.matched_vectors_linearfit_acc:
        v1 = np.dot(R_acc_orthogonalized, np.transpose(v1))
        repro_err_mag_acc.append(np.linalg.norm(v2 - v1))
        repro_err_ang_acc.append(utility.angle_between(v1, v2) * 180 / np.pi)
        repro_err_x_acc.append( v2[0] - v1[0] )
        repro_err_y_acc.append( v2[1] - v1[1] )
        repro_err_z_acc.append( v2[2] - v1[2] )
        
    xdata = range(len(rotationMatrix.matched_vectors_linearfit_acc))
    
    plot_repro_err_acc = plt.figure( num=next(figCounter), figsize=(10,10) )
    plot_repro_err_acc.suptitle( "ACC Graphs" )
    plt.subplot(511)
    plt.plot(xdata, repro_err_x_acc, color="#ff0000", label="x", linewidth=1)
    plt.legend()
    plt.subplot(512)
    plt.plot(xdata, repro_err_y_acc, color="#00ff00", label="y", linewidth=1)
    plt.legend()
    plt.subplot(513)
    plt.plot(xdata, repro_err_z_acc, color="#0000ff", label="z", linewidth=1)
    plt.legend()
    plt.subplot(514)
    plt.plot(xdata, repro_err_mag_acc, color='#943179', label="magnitude", linewidth=1)
    plt.legend()
    plt.subplot(515)
    plt.plot(xdata, repro_err_ang_acc, color='#ffa500', label="angle", linewidth=1)
    plt.legend()        
    
def plotReprojectionErrorGyro():
    """
    Calculates and plots the reprojection error using orthogonalized rotation 
    matrix from gyro data

    Returns
    -------
    None.

    """
    global R_gyro_orthogonalized, plot_repro_err_gyro, figCounter
    
    if (type(R_gyro_orthogonalized) == type(None)):
        orthogonalizeRotationMatrixGyro()
    
    repro_err_mag_gyro = []
    repro_err_mag_gyro = []
    repro_err_ang_gyro = []
    repro_err_x_gyro = []
    repro_err_y_gyro = []
    repro_err_z_gyro = []
    
    for v1,v2 in rotationMatrix.matched_vectors_linearfit_gyro:
        v1 = np.dot(R_gyro_orthogonalized, np.transpose(v1))
        repro_err_mag_gyro.append(np.linalg.norm(v2 - v1))
        repro_err_ang_gyro.append(utility.angle_between(v1, v2) * 180 / np.pi)
        repro_err_x_gyro.append( v2[0] - v1[0] )
        repro_err_y_gyro.append( v2[1] - v1[1] )
        repro_err_z_gyro.append( v2[2] - v1[2] )
        
    xdata = range(len(rotationMatrix.matched_vectors_linearfit_gyro))
    
    plot_repro_err_gyro = plt.figure( num=next(figCounter), figsize=(10,10) )
    plot_repro_err_gyro.suptitle( "GYRO Graphs" )
    plt.subplot(511)
    plt.plot(xdata, repro_err_x_gyro, color="#ff0000", label="x", linewidth=1)
    plt.legend()
    plt.subplot(512)
    plt.plot(xdata, repro_err_y_gyro, color="#00ff00", label="y", linewidth=1)
    plt.legend()
    plt.subplot(513)
    plt.plot(xdata, repro_err_z_gyro, color="#0000ff", label="z", linewidth=1)
    plt.legend()
    plt.subplot(514)
    plt.plot(xdata, repro_err_mag_gyro, color='#943179', label="magnitude", linewidth=1)
    plt.legend()
    plt.subplot(515)
    plt.plot(xdata, repro_err_ang_gyro, color='#ffa500', label="angle", linewidth=1)
    plt.legend()  
    
def plotReprojectionErrorBoth():
    """
    Calculates and plots the reprojection error using orthogonalized rotation 
    matrixes from acc and gyro data

    Returns
    -------
    None.

    """
    plotReprojectionErrorAcc()
    plotReprojectionErrorGyro()

            
            
            
            


















