from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def print_3d_graph_of(points, width, height, title) :
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    print('points',points)
    
    # (learning rate, batch size, accuracy)
    samples = [(p[0],p[1],p[2]) for p in points]
    
    ax.scatter(*zip(*samples))
    plt.title(title)
    
    ax.set_xlabel('learning rate')
    ax.set_ylabel('batch size')
    ax.set_zlabel('accuracy')
    
    plt.xlim(0,width)
    plt.ylim(0,height)
    
    plt.show(title)
    
def print_2d_graph_of(points, width, height, title) :
    fig = plt.figure()
    
    # (learning rate, batch size)
    samples = [(p[0],p[1]) for p in points]
    
    plt.scatter(*zip(*samples))
    plt.title(title)
    
    plt.xlabel('learning rate')
    plt.ylabel('batch size')
    
    plt.xlim(0,width)
    plt.ylim(0,height)
    
    plt.savefig(title)

def grab_tuple_from(file_array,i) :
    acc_str = file_array[i+1][-4:-2]
    point_str = file_array[i+2][12:-2]
    point_tuple = tuple(map(float, point_str.split(', ')))
    acc_tuple = tuple(map(float, acc_str.split(' ')))
    total_tuple = point_tuple + acc_tuple
    return point_tuple + acc_tuple

def print_final_results(filename):
    with open(filename,'r') as ins :
        file_array = []
        for line in ins :
            file_array.append(line)
    ins.close()

    network_type = ''
    random_points = []
    poisson_points = []
    automan_points = []

    for i in range(len(file_array)) :
        found = False
        line = file_array[i]
        print(line)
        if 'RANDOM EXPERIMENT DONE.\n' == line :
            network_type = 'RANDOM'
            random_points.append(grab_tuple_from(file_array,i))
        elif 'POISSON EXPERIMENT DONE.\n'==line :
            network_type = 'POISSON'
            poisson_points.append(grab_tuple_from(file_array,i))
        elif 'AUTO MANUAL EXPERIMENT DONE.\n'==line :
            network_type = 'AUTO'
            automan_points.append(grab_tuple_from(file_array,i))

    print_3d_graph_of(random_points,  1, 1000,
                   'Random Results From Affine Network') 

    print_3d_graph_of(poisson_points, 1, 1000,
                   'Poisson Results From Affine Network') 

    print_3d_graph_of(automan_points,  1, 1000,
                   'Auto Manual Results From Affine Network') 
    
    