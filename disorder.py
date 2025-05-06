import numpy as np
import scipy.sparse
from math import pi
import sys
import magnetic_boundaries as mb

job = int(sys.argv[1])

parameters = dict(
    Ly = 61,
    Lx = 61,
    theta_top = 0,
    theta_bot = 0,
    mag_field = 0.1,
    noise = 0,
    seed = job,
    mass = 0,
)
nbands = (parameters['Lx']*parameters['Ly'])//2

spectrum = mb.tan_square_spectrum(parameters, nbands)

path = '/home/donisvelaa/github/tangent-magnetic-field/data/'
name = 'disorder_tangent'
np.save(path+name+'_Lx'+str(parameters['Lx'])+'_Ly'+str(parameters['Ly'])+str(job), spectrum, allow_pickle=True)