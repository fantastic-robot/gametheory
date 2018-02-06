# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:46:19 2018

@author: Fanta
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:04:08 2018

@author: Fanta
"""
import pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import animation

from game_theory import *

def func(x, y, lik):
    return lik[x, y]


def plots(filename, title, xlim, ylim, zlim):
    
    max_loglik, loglik = fit_data_game(filename, xlim, ylim)
    
     # extract u_crash, u_time, likelihood from list of tuples as X, Y, Z
    X = np.array([x[0] for x in loglik])   # u_crash
    Y = np.array([x[1] for x in loglik])   # u_time
    Z = np.array([x[2] for x in loglik])   #likelihood
    
    if 1:
        print(X)
        print(Y)
        print(Z)
        
    if 1:
        print("stem plot")
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        for x, y, z in zip(X, Y, Z):        
            line=art3d.Line3D(*zip((x, y, 0), (x, y, z)), marker='o', markevery=(1, 1))
            ax.add_line(line)
        ax.set_xlim3d(-xlim, 0)
        ax.set_ylim3d(0, ylim)
        ax.set_zlim3d(-zlim, 0)
        ax.set_title(title + ', best parameters: ' + str(max_loglik))
        ax.set_xlabel("U_crash")
        ax.set_ylabel("u_time")
        ax.set_zlabel("LogLikelihood")
    
        plt.savefig(title + '_stem_plot.png', bbox_inches='tight')
        plt.show()
        
    if 1:
        
        matplotlib.rcParams['xtick.direction'] = 'out'
        matplotlib.rcParams['ytick.direction'] = 'out'
        
        # generate u_crash, u_time as X,  Y arrays of integers 
        X = np.arange(-xlim, 0)
        Y = np.arange(0, ylim)
        #X, Y = np.meshgrid(X, Y)
        
        print("contour plot")
        print(X.shape[0])
        print(Y.shape[0])
        print("shape loglik: " + str(np.shape(loglik)))
        
        lik = np.array([i[2] for i in loglik]) # extract likelihood from list of tuples
        print(loglik)
        print(np.shape(lik))
        lik = np.reshape(lik, (X.shape[0], Y.shape[0]))  # give shape (X,Y) to loglik
        print(lik)
        print(np.shape(lik))
        
        Z = np.zeros((X.shape[0], Y.shape[0]))  # give shape (X,Y) to loglik
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                Z[i,j] = func(i, j, lik)  # fill Z from loglik
        
        # Create a simple contour plot with labels using default colors.  The
        # inline argument to clabel will control whether the labels are draw
        # over the line segments of the contour, removing the lines beneath
        # the label
        figure()
        CS = plt.contour(Y, X, Z)
        clabel(CS, inline=1, fontsize=10)
        plt.title(title + ", optimum: " + str(max_loglik))         
        xlabel("U_crash")
        ylabel("U_time")
        #plt.xlim(-xlim, 0)
        #plt.ylim(0, ylim)
        #plt.zlim(-zlim, 0)
        show()
        
      
    if 1:
        plt.figure(3)
        
        print("Colormap plot")
        print("X shape" + str(X.shape))
        print("Y shape" + str(Y.shape))
        X,Y = np.meshgrid(X,Y)
        
        lik = np.array([i[2] for i in loglik]) # extract likelihood from list of tuples
        lik = np.reshape(lik, (X.shape[0], Y.shape[1]))  # give shape (X,Y) to loglik
        print(lik)
        print(np.shape(lik))
        
        Z = np.zeros((X.shape[0], Y.shape[1]))  # create empty array with X shape
        print("Shape z: " + str(np.shape(Z)))
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                Z[i,j] = func(i, j, lik)  # fill in Z with loglik values
    
        plt.title(title + ', optimum: ' + str(max_loglik))
        xlabel("U_crash")
        ylabel("U_time")
    
        # plot the calculated function values
        plt.pcolor(X,Y,Z)
        # and a color bar to show the correspondence between function value and color
        colorbar()
        show() 
        
        
        
if __name__ == "__main__":
    
    plots('natural_board_game_data.csv', 'Natural game', xlim=7, ylim=2, zlim=4000)
    #plots('chocolate_board_game_data.csv', 'Chocolate game',5, 3, 4000)
    #plots('board_game_data.csv', 'Mixed game', 5, 3, 4000)
    
    
    
    
