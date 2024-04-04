# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    def F(self):
        # TODO Step 1: implement and return system matrix F
        dt = params.dt
        #constant velocity 3D motion model
        F = np.matrix([[1,0,0,dt,0,0],
                       [0,1,0,0,dt,0],
                       [0,0,1,0,0,dt],
                       [0,0,0,1,0,0],
                       [0,0,0,0,1,0],
                       [0,0,0,0,0,1]])
        return F

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        dt = params.dt
        q = params.q
        q1 = dt * q
        q2 = ((dt**2)/2) * q
        q3 = ((dt**3)/3) * q

        Q = np.matrix([[q3,0,q2,0,0,0],
                       [0,q3,0,0,q2,0],
                       [0,0,q3,0,0,q2],
                       [q2,0,0,q1,0,0],
                       [0,q2,0,0,q1,0],
                       [0,0,q2,0,0,q1]])
        return Q

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        F = self.F()
        Q = self.Q()

        x = F*track.x
        P = F*track.P*F.transpose() + Q

        track.set_x(x)
        track.set_P(P)

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        H = meas.sensor.get_H(track.x)
        S = self.S(track,meas,H)
        gamma = self.gamma(track,meas) #residual
        I = np.identity(params.dim_state) #dim state = process model dimension

        K = track.P * H.transpose() * np.linalg.inv(S) #Kalman gain
        x = track.x + (K*gamma) #updated x
        P = (I - K*H) * track.P #updated covariance

        track.set_x(x) #saving updated x in function set_x
        track.set_P(P) #saving updated P in function set_P

        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        hx =  meas.sensor.get_hx(track.x)
        gamma = meas.z - hx
        return gamma

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        S = H * track.P * H.transpose() + meas.R
        return S
