from nn import runExperiment
from poisson_sampling import get_Poisson_Points
from auto_manual_search import run_auto_manual
import random
from results_grapher import print_2dgraph_of

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

    # keep trying different random radii until a Poisson batch is generated such that
    # the number of points in the batch is close to desired_num_points
    while (desired_num_points + 10) < best_overall_num_points :
            # generate a random radius between 10 and 50 with six significant digits
            r = random.randint(500000,2000000) / 100000
            #r = random.randint(100,200)
            #r = 100

            # grab the batch of poisson points with size closest to desired_num_points without going under
            best_points = None
            
            # one range copy must be passed in for each dimension
            grid = Grid(r, desired_num_points, 
                        learning_rate_range,
                        hidden_size_range)

            # Here I'm creating a seed for the sampling and then generating samples with Grid.poisson
            rand = (random.uniform(0, learning_rate_range),
                    random.uniform(0, hidden_size_range))
            points = grid.poisson(rand)

            # if this is the first run, just set best_points to first batch
            if best_points == None :
                best_points = points
            # else grab the batch with size closest to desired_num_points without going under
            elif desired_num_points <= len(points) and len(points) < len(best_points) :
                best_points = points

            if desired_num_points <= len(best_points) and len(best_points) < best_overall_num_points : 
                best_overall_num_points = len(best_points)
                best_overall_points = best_points
            #print('finished testing with r',r,', best_overall_num_points:',best_overall_num_points)
            #for point in best_overall_points :
                #print(point)
    return best_overall_points

##############################################################
################# MAIN METHOD ################################
##############################################################
# define hyperparameters ... using "hyper-hyperparameters"
desired_num_points = 100 # how many points we can "afford" to test for

learning_rate_range = 1
hidden_size_range = 1000

# create a file to write results to
results = open('results.txt','w')

# generate completely random points
random_points = generate_random_batch(desired_num_points, learning_rate_range, hidden_size_range)

# graph points
print_graph_of(random_points, learning_rate_range, hidden_size_range, title='Random Selection Under Unscaled Axis')

'''
# run experiment with randomly generated points
print('STARTING RANDOM EXPERIMENT')
results.write('STARTING RANDOM EXPERIMENT\n')
test_accuracy, best_point = runExperiment(random_points, desired_num_points, auto_manual='')

# print results 
print('RANDOM EXPERIMENT DONE.')
results.write('RANDOM EXPERIMENT DONE.\n')
print('test_accuracy: ' + str(test_accuracy) + '%\n')
results.write('test_accuracy: ' + str(test_accuracy) +'%\n')
print('best point',best_point)
results.write('best point ' + str(best_point) + '\n')
'''

# generate poisson sampling
print('STARTING POISSON EXPERIMENT')
results.write('STARTING POISSON EXPERIMENT\n')
poisson_points = get_Poisson_Points(0.75, desired_num_points, learning_rate_range, hidden_size_range)

# graph points
print_graph_of(poisson_points, learning_rate_range, hidden_size_range, title='Poisson Selection With Unscaled Axis')

'''
# run experiment with poisson generated points
test_accuracy, best_point = runExperiment(poisson_points, desired_num_points, auto_manual='')

# print results 
print('POISSON EXPERIMENT DONE.')
results.write('POISSON EXPERIMENT DONE.\n')
print('test_accuracy: ',test_accuracy,'%')
results.write('test_accuracy: ' + str(test_accuracy) + '%\n')
print('best point',best_point)
results.write('best point ' + str(best_point) + '\n')
'''

# run experiment using auto manual method
print('STARTING AUTO MANUAL EXPERIMENT')
results.write('STARTING AUTO MANUAL EXPERIMENT\n')
test_accuracy, best_point, points_tested = run_auto_manual(learning_rate_range, desired_num_points)

# graph points
print_graph_of(points_tested, learning_rate_range, hidden_size_range, title='Auto-Manual Selection')

# print results
print('AUTO MANUAL EXPERIMENT DONE')
results.write('AUTO MANUAL EXPERIMENT DONE\n')
print('test accuracy',test_accuracy)
results.write('test accuracy' + str(test_accuracy) + '\n')
print('best_point',best_point)
results.write('best_point' + str(best_point) + '\n')
