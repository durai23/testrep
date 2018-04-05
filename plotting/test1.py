#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 01:34:40 2018

@author: arasan
"""

import matplotlib.pyplot as plt
import numpy as np
#
#plt.figure(1)                # the first figure
#plt.subplot(211)             # the first subplot in the first figure
#plt.plot([1, 2, 3])
#plt.subplot(212)             # the second subplot in the first figure
#plt.plot([4, 5, 6])
#
#
#plt.figure(2)                # a second figure
#plt.plot([4, 5, 6])          # creates a subplot(111) by default
#
#plt.figure(1)                # figure 1 current; subplot(212) still current
#plt.subplot(211)             # make subplot(211) in figure1 current
#plt.title('Easy as 1, 2, 3') # subplot 211 title

#fig, ax = plt.subplots()
#ax.plot([2,4,6],[1,2,3])
#plt.show()
a=np.arange(0,5,0.1)
b=np.arange(0,5,0.1)
print a,b
#plt.plot(a,b,'r-.',label='lol')
#plt.title('lol')
#plt.legend()
#plt.show()

#fig = plt.figure()
#ax=fig.add_subplot(111)
#ax.plot(a,b)
#ax.set_
fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].set_title('Upper Left')
axes[0,1].set(title='Upper Right')
axes[1,0].set(title='Lower Left')
axes[1,1].set(title='Lower Right')

# To iterate over all items in a multidimensional numpy array, use the `flat` attribute
for ax in axes.flat:
    # Remove all xticks and yticks...
    ax.set(xticks=[], yticks=[])
    
plt.show()