"""
Script object: fitting some data to the game theory model 
Author: Fanta Camara
Date: 17/01/2018
%dewbug"""


import numpy as np
import GPy
import pandas  as pd
from operator import itemgetter
from game import *
from script_board_game_data import *

def fit_data_game(filename, u_crash, u_time):
    
    """
    data: all the board game data from csv%debug
    file
    game = each line of data
    turn and turn_int = each move of the player, columns from 4 to 11  : [speedM destinationM speedU destinationU]
    U_crash: probability of a crash for both M and U
    U_time: parameter for time delay
    """
    
    data = numpy_matrix_from_csv(filename)
    log_likelihood = []		# log_likelihood values for all U_crash and U_time as a list of tuples (U_crash, U_time, likelihood)
    
    for U_crash in range(-u_crash, 0):
        for U_time in range(0, u_time):  # can't have float with "for in range"
            #Compute the strategy matrix (NM*NU*2), contains P( player X  chooses speed 2 | m,u)  (using function from existing code)
            
            U_crash = U_crash+0.0
            U_time = U_time +0.0
            
            (V,S) = solveGame(U_crash, U_crash, U_time, NY=20, NX=20)
            
            print("V matrix")
            print(len(V))
            print(np.shape(V))
            print(V)
            
            print("strategy matrix")
            print(len(S))
            print(np.shape(S))
            print(S)
            
            print("U_crash")
            print(U_crash)
            print("U_time")
            print(U_time)
            
            				# likelihood term
            S_noise = np.ones((20, 20, 2))/100
            
            log_lik_all_game = 0
            
            for game in data:
                print("\nNew Game")

                log_lik_game = 0
                
                for turn in game[4:11]:
                    
                    turn = np.array([int(x) for x in turn.split()])
                    print("New Turn")
                    print(turn)
                    
                    log_lik_turn = 0
                    
                    if(len(turn) > 0 and (turn[1] > -1 and turn[3] > -1)):  # if the turn has taken place, i.e sometimes turn 6 and/or 7 is empty
                    #log_lik += log  P(d^{game,turn}_{Y} | y, x,theta,M )    # read from strategy matrix for player 1
                        
                        print("yield prob X: " + str(S[turn[1], turn[3], 0]))
                        lik_X = 99/100*S[turn[1], turn[3], 0] + 1/100*S_noise[turn[1], turn[3], 0]
                                                   
                        if(lik_X > 1):
                            pdb.set_trace()
     
                        print("yield prob Y: " + str(S[turn[1], turn[3], 1]))
                        lik_Y = 99/100*S[turn[1], turn[3], 1] + 1/100*S_noise[turn[1], turn[3], 1]
                        lik_both = lik_X*lik_Y
                            
                        if(lik_X > 1):
                            pdb.set_trace()
     
                        
                        log_lik_turn = np.log(lik_both)
                    
                        print("strategy values for players 1 & 2")
                        print("p Player Y: " + str(lik_X))
                        print("p Player X: " + str(lik_Y))
                        print("loglik turn " + str(log_lik_turn))
                     
                        log_lik_game+= log_lik_turn
                        print("game loglik = " + str(log_lik_game))
                    
                log_lik_all_game += log_lik_game
                        
            #store log_likelihood(U_crash, Alpha) as a list of tuples
            log_likelihood.append((U_crash, U_time, log_lik_all_game))
            print("all game log_lik" + str(log_lik_all_game))
            
    print("\nList of log likelihood:")
    print(log_likelihood)
    
    #find arg_{U_crash, Alpha)}max  loglik(U_crash, Alpha)
    max_log_likelihood = max(log_likelihood, key=itemgetter(2))
    
    return max_log_likelihood, log_likelihood


if __name__ == "__main__":


    maxlik, loglik = fit_data_game("natural_board_game_data.csv", 10, 2) 
    print("loglik = " + str(loglik))
    print("maxlik = " + str(maxlik))
    

    """
    max_log_likelihood, log_likelihood = fit_data_game('board_game_data', 20, 5)
    print("\nMax log likelihood value\n")
    print(max_log_likelihood)
    
    U_crash = np.arange(-10, 0, 1)
    print(U_crash)
    
    print(log_likelihood)
    print(len(log_likelihood))
    print(np.shape(log_likelihood))
    
    
    x = np.arange(0,5,0.1).reshape(50,1)
    y = np.sin(x).reshape(50,1)
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(x,y, kernel)
    m.optimize()
    m.plot()
    
    
    x = np.arange(0,5,0.1).reshape(50,1)
    y = np.sin(x).reshape(50,1) + np.random.randn(50,1)*0.05
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    m = GPy.models.GPRegression(x,y, kernel)
    m.optimize()
    m.plot()
    
    
    # Plot !D Gaussian Process
    kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
    #kernel = GPy.kern.Linear(np.array([x[0] for x in log_likelihood]).reshape(20,1).shape[1], ARD=1)
    m=GPy.models.GPRegression(np.array([x[0] for x in log_likelihood]).reshape(200,1), np.array([x[1] for x in log_likelihood]).reshape(200,1), kernel)
    m.optimize()
    m.plot()
    
    print(np.array([x[0] for x in log_likelihood]).reshape(200,1))
    print(np.array([x[2] for x in log_likelihood]).reshape(200,1))
    
    """
    
    """
    # Plot 2D Gaussian Processes
    ker = GPy.kern.Matern52(2, ARD=True) + GPy.kern.White(2)
    m = GPy.models.GPRegression(U_crash, log_likelihood, ker)
    m.optimize(messages=True, max_f_eval = 1000)
    m.plot()
    """ 
	
