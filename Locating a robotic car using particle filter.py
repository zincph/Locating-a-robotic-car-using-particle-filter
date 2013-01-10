# Locating a robotic car using particle filter


# the "world" has 4 landmarks.
landmarks  = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]] # NOTE: Landmark coordinates are given in (y, x) form and NOT
# in the traditional (x, y) format!
# The robot's initial coordinates are somewhere in the square
# represented by the landmarks.
world_size = 100.0 # world is NOT cyclic. Robot is allowed to travel "out of bounds"

# The robot can move, and sense its own locations by measuring its bearings to the landmarks.
# However, in the real world, there are considerable noises with both move and sense functions.

from math import *
import random

max_steering_angle = pi / 4.0 # Won't be used, just the limitations of a real car.
bearing_noise = 0.1 # Noise parameter: included in sense function.
steering_noise = 0.1 # Noise parameter: included in move function.
distance_noise = 5.0 # Noise parameter: included in move function.

# The GOAL is to locate the car (position and orientation) within the following tolerances at least 80% of the time.

tolerance_xy = 15.0 # Tolerance for localization in the x and y directions.
tolerance_orientation = 0.25 # Tolerance for orientation.




# ------------------------------------------------
# 
# this is the robot class
#

class robot:

    # --------
    # init: 
    #    creates robot and initializes location/orientation 
    #

    def __init__(self, length = 20.0):
        self.x = random.random() * world_size # initial x position
        self.y = random.random() * world_size # initial y position
        self.orientation = random.random() * 2.0 * pi # initial orientation
        self.length = length # length of robot
        self.bearing_noise  = 0.0 # initialize bearing noise to zero
        self.steering_noise = 0.0 # initialize steering noise to zero
        self.distance_noise = 0.0 # initialize distance noise to zero

    # --------
    # set: 
    #    sets a robot coordinate
    #

    def set(self, new_x, new_y, new_orientation):

        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError, 'Orientation must be in [0..2pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    # --------
    # set_noise: 
    #    sets the noise parameters
    #
    def set_noise(self, new_b_noise, new_s_noise, new_d_noise):
        # makes it possible to change the noise parameters
        
        self.bearing_noise  = float(new_b_noise)
        self.steering_noise = float(new_s_noise)
        self.distance_noise = float(new_d_noise)

    # --------
    # measurement_prob
    #    computes the probability of a measurement
    #  

    def measurement_prob(self, measurements):

        # calculate the correct measurement
        predicted_measurements = self.sense(0) # sense function took 0 as an argument to switch off noise.


        # compute errors
        error = 1.0
        for i in range(len(measurements)):
            error_bearing = abs(measurements[i] - predicted_measurements[i])
            error_bearing = (error_bearing + pi) % (2.0 * pi) - pi # truncate
            

            # update Gaussian
            error *= (exp(- (error_bearing ** 2) / (self.bearing_noise ** 2) / 2.0) /  
                      sqrt(2.0 * pi * (self.bearing_noise ** 2)))

        return error
    
    def __repr__(self): #allows us to print robot attributes.
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), 
                                                str(self.orientation))

       
    # --------
    # move: 
    #
    #   It simulates motion noise
    #   according to the noise parameters
    #           self.steering_noise
    #           self.distance_noise
    #   Actual steering angle is chosen from a Gaussian
    #   distribution of steering angles. This distribution
    #   is centered at the intended steering angle
    #   with variance of self.steering_noise.
    
    def move(self, motion): 
        beta=(motion[1] + random.gauss(0.,self.distance_noise)) / self.length * tan(motion[0] + random.gauss(0.,self.steering_noise))

        if (beta>-0.001 and beta<0.001):
            x=self.x + (motion[1]+random.gauss(0.,self.distance_noise))*cos(self.orientation)
            y=self.y + (motion[1]+random.gauss(0.,self.distance_noise))*sin(self.orientation)
            orientation=self.orientation
        else:
            radius=self.length/tan(motion[0]+random.gauss(0.,self.steering_noise))
            cx=self.x-sin(self.orientation)*radius
            cy=self.y+cos(self.orientation)*radius
            x=cx+sin(self.orientation+beta)*radius
            y=cy-cos(self.orientation+beta)*radius
            orientation=(self.orientation+beta)%(2*pi)
            
        result=robot(self.length)
        result.set(x,y,orientation)
        result.set_noise(self.bearing_noise, self.steering_noise, self.distance_noise)
        
        
        return result 
      
  

    # --------
    # sense: 
    #
    #   The way to simulate bearing noise in the sense step is to
    #   add Gaussian noise, centered at zero with variance
    #   of self.bearing_noise to each bearing.
    #   This is done with the command random.gauss(0, self.bearing_noise)
    
    def sense(self,flag=1): 
        Z = []
        if flag==0:
            
            for i in range(len(landmarks)):
                Z.append((atan2((landmarks[i][0]-self.y),(landmarks[i][1]-self.x))-self.orientation)%(2*pi))
        elif flag==1:
            for i in range(len(landmarks)):
                Z.append(((atan2((landmarks[i][0]-self.y),(landmarks[i][1]-self.x))-self.orientation)%(2*pi))+random.gauss(0.,self.bearing_noise))



        return Z #Return vector Z of 4 bearings.



# --------
#
# extract position from a particle set
# 

def get_position(p):
    x = 0.0
    y = 0.0
    orientation = 0.0
    for i in range(len(p)):
        x += p[i].x
        y += p[i].y
        # orientation is tricky because it is cyclic. By normalizing
        # around the first particle we are somewhat more robust to
        # the 0=2pi problem
        orientation += (((p[i].orientation - p[0].orientation + pi) % (2.0 * pi)) 
                        + p[0].orientation - pi)
    return [x / len(p), y / len(p), orientation / len(p)]

# --------
#
# The following code generates the measurements vector
 


def generate_ground_truth(motions):

    myrobot = robot()
    myrobot.set_noise(bearing_noise, steering_noise, distance_noise)

    Z = []
    T = len(motions)

    for t in range(T):
        myrobot = myrobot.move(motions[t])
        Z.append(myrobot.sense())
    
    return [myrobot, Z]

# --------
#
# The following code prints the measurements associated
# with generate_ground_truth
#

def print_measurements(Z):

    T = len(Z)

    print 'measurements = [[%.8s, %.8s, %.8s, %.8s],' % \
        (str(Z[0][0]), str(Z[0][1]), str(Z[0][2]), str(Z[0][3]))
    for t in range(1,T-1):
        print '                [%.8s, %.8s, %.8s, %.8s],' % \
            (str(Z[t][0]), str(Z[t][1]), str(Z[t][2]), str(Z[t][3]))
    print '                [%.8s, %.8s, %.8s, %.8s]]' % \
        (str(Z[T-1][0]), str(Z[T-1][1]), str(Z[T-1][2]), str(Z[T-1][3]))

# --------
#
# The following code checks to see if the particle filter
# localizes the robot to within the desired tolerances
# of the true position. The tolerances are defined at the top.
#

def check_output(final_robot, estimated_position):

    error_x = abs(final_robot.x - estimated_position[0])
    error_y = abs(final_robot.y - estimated_position[1])
    error_orientation = abs(final_robot.orientation - estimated_position[2])
    error_orientation = (error_orientation + pi) % (2.0 * pi) - pi
    correct = error_x < tolerance_xy and error_y < tolerance_xy \
              and error_orientation < tolerance_orientation
    return correct



def particle_filter(motions, measurements, N=500): 
    # --------
    #
    # Make particles
    # 

    p = []
    for i in range(N):
        r = robot()
        r.set_noise(bearing_noise, steering_noise, distance_noise)
        p.append(r)

    # --------
    #
    # Update particles
    #     

    for t in range(len(motions)):
    
        # motion update (prediction)
        p2 = []
        for i in range(N):
            p2.append(p[i].move(motions[t]))
        p = p2

        # measurement update
        w = []
        for i in range(N):
            w.append(p[i].measurement_prob(measurements[t]))

        # resampling
        p3 = []
        index = int(random.random() * N)
        beta = 0.0
        mw = max(w)
        for i in range(N):
            beta += random.random() * 2.0 * mw
            while beta > w[index]:
                beta -= w[index]
                index = (index + 1) % N
            p3.append(p[index])
        p = p3
    
    return get_position(p)

## IMPORTANT: Uncomment the test cases below to test the code.
##
## One way to test whether the particle filter works is to use the
## function check_output (see test case 2).
## Note: Even for a well-implemented particle filter this
## function occasionally returns False. This is because a particle
## filter is a randomized algorithm. 

## GOAL: check_output returns True at least 80% of the time.
## 


 
## --------
## TEST CASES:
## 
## 1) Calling the particle_filter function with the following
##    motions and measurements should return a [x,y,orientation]
##    vector near [x=93.476 y=75.186 orient=5.2664], that is, the
##    robot's true location.
 
# motions = [[2. * pi / 10, 20.] for row in range(8)]
# measurements = [[4.746936, 3.859782, 3.045217, 2.045506],
#                [3.510067, 2.916300, 2.146394, 1.598332],
#                [2.972469, 2.407489, 1.588474, 1.611094],
#                [1.906178, 1.193329, 0.619356, 0.807930],
#                [1.352825, 0.662233, 0.144927, 0.799090],
#                [0.856150, 0.214590, 5.651497, 1.062401],
#                [0.194460, 5.660382, 4.761072, 2.471682],
#                [5.717342, 4.736780, 3.909599, 2.342536]]
#
# print particle_filter(motions, measurements)

## 2) You can generate your own test cases by generating
##    measurements using the generate_ground_truth function.
##    It will print the robot's last location when calling it.
##
 
# number_of_iterations = 6
# motions = [[2. * pi / 20, 12.] for row in range(number_of_iterations)]

# x = generate_ground_truth(motions)
# final_robot = x[0]
# measurements = x[1]
# estimated_position = particle_filter(motions, measurements)
# print_measurements(measurements)
# print 'Ground truth:    ', final_robot
# print 'Particle filter: ', estimated_position
# print 'Code check:      ', check_output(final_robot, estimated_position)



