#NB  val(y,x,b)=v(x,y,notb)  for the symetric game

def ymax(v1,v2):    #given two worlds with player values, which is best for Y?
	if v1[0]>v2[0]:
		return v1
	else:
		return v2

def xmax(v1,v2):     #which is best for X
	if v1[1]>v2[1]:
		return v1
	else:
		return v2

def yargmax(w1,w2):     #world vectors
	if world_value(w1)[0] > world_value(w2)[0]:
		return w1
	else:
		return w2

def xargmax(w1,w2):     #world vectors
	if world_value(w1)[1] > world_value(w2)[1]:
		return w1
	else:
		return w2


def world_value(world):   #mydisttox, yorudisttox, timelapsed, bmyturn).  value = -T taken to reach x.
	(y,x,t,b) = world
	U_crash =-20
	U_time  = 1.0

	#end states
	if x==y==0:
		return (U_crash, U_crash)
	if x==y==1:
		return (U_crash, U_crash)

	if y==0:
		#return ( 0., -U_time*x/2. )
		return (-U_time*t , -U_time*t - U_time*x/2.)
	if y==1:
		#return ( 0., -U_time*(x-1)/2. )
		return (-U_time*(t+.5) , -U_time*t - U_time*(x-1)/2.)
	if x==0:
		#return ( -U_time*y/2.  , 0.)
		return ( -U_time*t - U_time*y/2.,  -U_time*t)
	if x==1:
		#return ( -U_time*(y-1)/2.  , 0.)
		return ( -U_time*(t) - U_time*(y-1)/2.,  -U_time*(t+.5))

	if x<0 or y<0:
		print("ERROR")
		pdb.set_trace()

	if b:    #b=it is Y's turn
		return ymax( world_value((y-2,x,t+1,False)), \
			      world_value((y-1,x,t+1,False)) )
	if not b:
		return xmax( world_value((y, x-2, t+1, True)), \
				world_value((y, x-1, t+1, True)) ) 			
def act(world):
	(y,x,t,b)=world
	if b:
		return yargmax( (y-2,x,t+1,False), \
			        (y-1,x,t+1,False) )
	if not b:
		return xargmax( (y, x-2, t+1, True), \
				(y, x-1, t+1, True) ) 			

import pdb
import numpy as np
from pylab import *

vv = world_value((2,2,0,True))

NY=12
NX=12
VY = np.zeros((NY,NX))
VX = np.zeros((NY,NX))
for y in range(0,NY):
	for x in range(0,NX):
		vY = world_value((y,x,0,True))
		VY[y,x] = vY[0]
		vX = world_value((y,x,0,False))
		VX[y,x] = vX[1]

figure(1)
imshow(VY, interpolation='none', cmap='gray')
xlabel("y, player Y location (meters)")
ylabel("x, player X location (meters)")
title("Value to Y, player Y to play")

figure(2)
imshow(VX, interpolation='none', cmap='gray')
xlabel("y, player Y location (meters)")
ylabel("x, player X location (meters)")
title("Value to X, player X to play")


#simulate
log=[]
world = (10,8,0, True)
t=0
b_done=0
while not b_done:
	log.append(world)
	world=act(world)

	print(world)
	
	(y,x,t,b)=world
#	if x<2 or y<2:
#		b_done=True
#		log.append(world)
	if t>9:
		b_done = True

ms = [ world[0] for world in log]
us = [ world[1] for world in log]
ts = [ world[2] for world in log]

figure(3)
hold(1)
plot(ts,ms, 'k')
plot(ts,us, 'k--')
legend(['y, player Y position (meters)', 'x, player X position (meters)'])
plot( ts, [0 for t in ts], 'k' )
xlabel('time')
ylabel('vehicle location')
title('Simulated trajectories')
show()		
