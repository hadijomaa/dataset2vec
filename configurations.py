#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:56:03 2020

@author: hsjomaa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:16:13 2020

@author: hsjomaa
"""

import json
import os
# setup default configuration
configuration = {}
# data shape
configuration['nonlinearity_d2v'] = None
# Function F
configuration['units_f']  = None
configuration['nhidden_f']  = None
configuration['architecture_f']  = "SQU"
configuration['resblocks_f'] = 8
        # Function H
configuration['units_h']  = None
configuration['nhidden_h']  = None
configuration['architecture_h']  = None
configuration['resblocks_h'] = 8

        # Function G
configuration['units_g']  = None
configuration['nhidden_g']  = None
configuration['architecture_g']  = None

configuration['ninstanc'] = 256
configuration['nclasses'] = 5
configuration['nfeature'] = 32
        
# initialize 
rootdir     = os.path.dirname(os.path.realpath(__file__))
counter = 0
# iterate over nonlinearities:
for nonlinearity_d2v in ['relu']:
    # iterate over learning_rate
        configuration['nonlinearity_d2v'] = nonlinearity_d2v
        for nhidden_f in [4]:
            configuration['nhidden_f'] = nhidden_f
            for nhidden_g in [4]:
                    configuration['nhidden_g'] = nhidden_g            
                    for nhidden_h in [4]:
                                configuration['nhidden_h'] = nhidden_h
                                for units_f in [32]:
                                    configuration['units_f'] = units_f
                                    for units_g in [32]:
                                        configuration['units_g'] = units_g                                
                                        for units_h in [32]:
                                                configuration['units_h'] = units_h
                                                for architecture_f in ['SQU']:
                                                    configuration['architecture_f'] = architecture_f
                                                    for architecture_g in ['SQU']:
                                                        configuration['architecture_g'] = architecture_g
                                                        for architecture_h in ['SQU']:
                                                            configuration['architecture_h'] = architecture_h
                                                            configuration['number'] = counter
                                                            json.dump(configuration,open(os.path.join(rootdir,"configurations",f"configuration-{counter}.json"),'w'))
                                                            counter+=1