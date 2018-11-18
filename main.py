from nn import runExperiment
from poisson_sampling import getPoints
import random

# GENERATE RANDOM BATCH
# function generates a list of tuples of size desired_num_points, each
# of which are randomly generated values in the ranges learning_rate_range
# and hidden_size_range
# IN: the following are all integers: desired_num_points , learning_rate_range, hidden_size_range
# OUT: a list of tuples representing point coordinates in hyperparameter space
def generate_random_batch(desired_num_points, learning_rate_range, hidden_size_range) :
    points = [] # initialize empty point list
    # generate desired_num_points points
    for i in range(desired_num_points):
        # learning rate will have seven significant digits
        lr = random.randint(0,learning_rate_range * 1000000) / 1000000
        hs = random.randint(0,hidden_size_range)
        
        points.append((lr,hs))
    return points

# GENERATE POISSON BATCH
# this function will generate a poisson disc sampling of size desired_num_points within a range
# specified by learning_rate_range and hidden_size_range
# IN: the following are all integers: desired_num_points , learning_rate_range, hidden_size_range
# OUT: a list of tuples representing point coordinates in hyperparameter space
def generate_Poisson_batch(desired_num_points, learning_rate_range, hidden_size_range) :
    # keep generating poisson sample batches until a batch is 
    # found with at most desired_num_points + 10 points in it
    best_overall_num_points = 200000
    best_overall_points = None

    while (desired_num_points + 10) < best_overall_num_points :
        # generate a random radius between 0 and 2 with six significant digits
        r = random.randint(0,200000) / 100000

        # grab the batch of poisson points with size closest to desired_num_points without going under
        best_points = None
        for i in range(5) :
            points = getPoints(r, desired_num_points, learning_rate_range, hidden_size_range)    

            # if this is the first run, just set best_points to first batch
            if best_points == None :
                best_points = points
            # else grab the batch with size closest to desired_num_points without going under
            elif desired_num_points <= len(points) and len(points) < len(best_points) :
                best_points = points

        if desired_num_points <= len(best_points) and len(best_points) < best_overall_num_points : 
            best_overall_num_points = len(best_points)
            best_overall_points = best_points

    return best_overall_points

##############################################################
################# MAIN METHOD ################################
##############################################################
# define hyperparameters ... using "hyper-hyperparameters"
learning_rate_range = 1
hidden_size_range = 1000
desired_num_points = 25 # how many points we can "afford" to test for

# generate completely random points
random_points = generate_random_batch(desired_num_points, learning_rate_range, hidden_size_range)

# run experiment with randomly generated points
print('STARTING RANDOM EXPERIMENT')
test_accuracy, best_learning_rate, best_hidden_size = runExperiment(random_points, desired_num_points)

# print results 
print('RANDOM EXPERIMENT DONE.')
print('test_accuracy: ',test_accuracy,'%')
print('best_learning_rate',best_learning_rate)
print('best_hidden_size',best_hidden_size)

# generate poisson sampling
print('STARTING POISSON EXPERIMENT')
poisson_points = generate_Poisson_batch(desired_num_points, learning_rate_range, hidden_size_range)

# run experiment with randomly generated points
test_accuracy, best_learning_rate, best_hidden_size = runExperiment(poisson_points, desired_num_points)

# print results 
print('POISSON EXPERIMENT DONE.')
print('test_accuracy: ',test_accuracy,'%')
print('best_learning_rate',best_learning_rate)
print('best_hidden_size',best_hidden_size)
