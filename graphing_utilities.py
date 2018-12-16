import numpy as np
import matplotlib.pyplot as plt

# Note that this only graphs the first two dimensions of points
def print_graph_of(points, width, height, title) :
    #plt.plot(samples[0], samples[1:])
    print('points',points)
    # only use the first two elements of each tuple in points
    samples = [(p[0],p[1]) for p in points]
    
    plt.scatter(*zip(*samples))
    plt.title(title)
    plt.xlim(0,width)
    plt.ylim(0,height)
    plt.show(title)
