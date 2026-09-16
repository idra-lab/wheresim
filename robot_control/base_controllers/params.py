# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np

robot_params = {}

robot_params['solo'] ={'dt': 0.002,
                       'kp': [5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5.],
                       'kd': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       'q_0':  np.array([0,  np.pi/4, -np.pi/2,    0, -np.pi/4,  np.pi/2, -0,  np.pi/4, -np.pi/2, -0, -np.pi/4,  np.pi/2]),
                       'q_fold': np.array([0,  1.57, -3.13, 0,    1.57, -3.13, 0, -1.57, 3.13, 0, -1.57, 3.13]),
                       'joint_names': ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint',
                                       'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint',
                                       'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint',
                                       'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint'],
                       'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                       'real_robot': False,
                       'force_th': 2,
                       'spawn_x': 0.0,
                       'spawn_y': 0.0,
                       'spawn_z': 0.3,
                       'buffer_size': 1501} # note the frames are all aligned with base for joints = 0

robot_params['aliengo'] ={'dt': 0.002,
                          'kp': np.array([60., 90., 60.]*4),
                          'kd': np.array([10., 10., 10.]*4),
                          'ki': np.array([0., 0., 0.]*4),
                           # joint pid + wbc (optional)
                          'kp_wbc': np.array([60., 60., 60.]*4),
                          'kd_wbc': np.array([10., 10., 10.]*4),
                          'ki_wbc': np.array([0., 0., 0.]*4),
                          # virtual impedance wrench control
                          'kp_lin': np.array([1000., 1000., 800.]),
                          'kd_lin': np.array([150., 150., 100.]),
                          'kp_ang': np.array([200., 200., 100.]),
                          'kd_ang': np.array([10., 20., 20.]),

                          #orbit
                          'q_0': np.array([0.0951, 0.8303,-1.5419,
                                           0.0980, 0.9864, -1.4778,
                                           -0.0948, 0.8305,-1.5420,
                                           -0.0979,0.9864, -1.4779]),
                          'q_fold': np.array([0.2, 1.7, -2.7, 0.2, 1.7, -2.7, -0.2, 1.7, -2.7, -0.2, 1.7, -2.7]),  # thjis is for the startup phase

                         'joint_names': ['lf_haa_joint', 'lf_hfe_joint', 'lf_kfe_joint',
                                       'lh_haa_joint', 'lh_hfe_joint', 'lh_kfe_joint',
                                       'rf_haa_joint', 'rf_hfe_joint', 'rf_kfe_joint',
                                       'rh_haa_joint', 'rh_hfe_joint', 'rh_kfe_joint'],
                        'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                        'real_robot': False,
                        'force_th': 10.,
                        'spawn_x': 0.0,
                        'spawn_y': 0.0,
                        'spawn_z': 0.37,
                        'ip': "192.168.123.220",
                        'buffer_size': 50001}

robot_params['go2'] ={'dt': 0.002,
                      'buffer_size': 25001, # 120 seconds

                      'kp': np.array([60., 60., 60.] * 4),
                      'kd': np.array([0.8, 0.8, 0.8] * 4),
                      'ki': np.array([0., 0., 0.] * 4),

                      # joint pid + wbc (optional)
                      'kp_wbc': np.array([15., 15., 15.]*4),
                      'kd_wbc': np.array([1., 1., 1.]*4),
                      'ki_wbc': np.array([0., 0., 0.]*4),
                      # virtual impedance wrench control
                      'kp_lin': np.array([800, 500., 900.]),  # x y z
                      'kd_lin': np.array([100, 100., 100.]),
                      'kp_ang': np.array([40., 40., 40.]),  # R P Y
                      'kd_ang': np.array([1.51, 1.51, 1.51]),                      
                       'q_0':  np.array([0.1, 0.8, -1.5,#lf
                                         0.1, 1.0, -1.5,#lh
                                         -0.1, 0.8, -1.5,#rf
                                         -0.1, 1.0, -1.5]),#rh
                      'q_fold': np.array([0.2, 1.7, -2.7, 0.2, 1.7, -2.7, -0.2, 1.7, -2.7, -0.2, 1.7, -2.7]),
                      'joint_names': ['lf_haa_joint',  'lf_hfe_joint', 'lf_kfe_joint',
                                      'lh_haa_joint',  'lh_hfe_joint', 'lh_kfe_joint',
                                      'rf_haa_joint',  'rf_hfe_joint', 'rf_kfe_joint',
                                      'rh_haa_joint',  'rh_hfe_joint', 'rh_kfe_joint'],
                      # ee params
                      'ee_frames': ['lf_foot', 'lh_foot', 'rf_foot','rh_foot'],
                      'force_th': 7.,
                      # simulation spawn [m] and [rad]
                      'spawn_x': 0.0,
                      'spawn_y': 0.0,
                      'spawn_z': .32,
                      'spawn_R': 0.0,
                      'spawn_P': 0.0,
                      'spawn_Y': 0.0,
                      'ip': "192.168.123.161",
                      # use real robot or simulation
                      'real_robot': False}


verbose = False
plotting = True

