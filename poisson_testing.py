from matplotlib import pyplot as plt
import seaborn
from math import sqrt, pi
from poisson_disc_generalized import Grid
import random


# keep generating poisson sample batches until a batch is 
# found with at most desired_num_points + 10 points in it
best_overall_num_points = 200000
best_overall_points = None
desired_num_points = 10

while (desired_num_points + 10) < best_overall_num_points :
        # generate a random radius between 10 and 50 with six significant digits
        #r = random.randint(500000,2000000) / 100000
        r = random.randint(100,200)
        #r = 100

        # grab the batch of poisson points with size closest to desired_num_points without going under
        best_points = None
        for i in range(2) :
            length = 1 # learning rate
            width = 1000 # hidden size
            height = 1000 # batch size
            desired_num_points = 10

            grid = Grid(r, desired_num_points, length, width, height)

            # Here I'm creating a seed for the sampling and then generating samples with Grid.poisson
            rand = (random.uniform(0, length), random.uniform(0, width), random.uniform(0,height))
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
print('found a suitable batch of size',best_overall_num_points)

for point in best_overall_points :
    print(point)