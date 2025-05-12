import numpy as np
import scipy.sparse
from math import pi
import sys
import magnetic_boundaries as mb

job = int(sys.argv[1])

parameters = {
    'width' : 151, # This is in units of 3a
    'length' : 51,# This is in units of sqrt(3)a
    'bottom_bearded' : True,
    'top_bearded' : False,
    'mag_field' : 0.1,
    'noise' : 0.1,
    'seed' : job
             }

spectrum = mb.graphene_spectrum(parameters)

path = '/home/donisvelaa/github/tangent-magnetic-field/data/'
name = 'disorder_graphene'
np.save(path+name+'_length'+str(parameters['length'])+'_width'+str(parameters['width'])+'_magfield'+str(parameters['mag_field'])+'_noise'+str(parameters['noise'])+str(job), spectrum, allow_pickle=True)
