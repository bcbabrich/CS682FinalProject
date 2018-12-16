import random
import math
from random import shuffle
from nn import runExperiment

# GET PERTURB 
# returns a distance along a single axis to "perturb" a point by
# IN: a point along an axis (int, best_h) and a counter value (int, k)
# OUT: a distance along the axis which is proportional to k (int)
def get_perturb(best_h, k) :
    # get order of magnitude of best_h
    ord_mag = int(math.log10(best_h))
    
    # divide accordingly to get integer
    starting_p = int(best_h/(10 ** (ord_mag)))
    
    # scale starting_p using k
    #final_p = starting_p * k # narrowest search
    final_p = starting_p * int(k**math.log(k)) # narrower search
    #final_p = starting_p ** k # widest search
    
    # add small value to prevent repeat p returns
    final_p += random.randint(0,int(best_h/(10 ** (ord_mag))))
    
    # assign perturb value random sign
    signs = [-1,1]
    final_p *= random.choice(signs)
    
    return final_p

# returns a point with current dictionary vales
# helper method
# IN: a dictionary mapping hyper-parameter names to int values
# OUT: a tuplet representing a point generated using the dictionary
def def_point(param_dict) :
    point = (param_dict['lr'], 
             param_dict['hs'], 
             param_dict['bs'], 
             param_dict['ne'])
    return point

# simulates an experiment run
# returns a random "accuracy" percentage
# not used in actual experiment, useful for debugging
def faux_experiment(point) :
    return random.randint(0,100)

# main
def run_auto_manual(hyper_parameter_range, desired_num_points, network_type, experiment_number) :
    printHelp = False
    points_tested = []
    
    if printHelp : print('hyper_parameter_range',hyper_parameter_range)
    
    # draw a random point from hyper-parameter space
    lr = random.randint(0,hyper_parameter_range) # learning rate 
    hs = random.randint(0,hyper_parameter_range) # hidden size
    bs = random.randint(0,hyper_parameter_range) # batch size
    ne = random.randint(0,hyper_parameter_range) # number of epochs
    
    # we need a dictionary and a list because we want to return
    # the hyperparameters in a predefined order but we want to
    # search through them in a random order
    param_dict = {}
    param_dict['lr'] = lr
    param_dict['hs'] = hs
    param_dict['bs'] = bs
    param_dict['ne'] = ne
    
    active_parameters = ['lr','hs','bs','ne'] # predefined order
    shuffle(active_parameters) # random order
    
    if printHelp : print('active_parameters (shuffled)',active_parameters)
    
    # the number of times we can try adjusting a single hyper-parameter
    num_trials = int(desired_num_points/len(active_parameters))
    
    if printHelp : print('num_trials',num_trials)
    
    for h in active_parameters :
        if printHelp : print('current param being adjusted',h)
        best_acc = 0
        best_h = param_dict[h]
        k = 1
        
        for t in range(num_trials) :
            if printHelp : print('best ',h,' found so far',best_h,', gave an acc of ',best_acc)
            # define point using current values in param_dict
            point = def_point(param_dict)
            
            # run experiment with point
            acc = faux_experiment(point)
            #print('running experiment using automanual point',point)
           # acc, experiment_number = runExperiment([point], 1, 'val', network_type, experiment_number)
            points_tested.append(point) # keep track of all points for graphing purposes
            # print('accuracy returned was',acc)
            if printHelp : print('when ran with ',h,' = ',param_dict[h],' acc',acc)
            
            # get perturb value
            # this will grow with the number of consecutive failed attempts
            p = get_perturb(best_h, k)
            if printHelp : print('p',p)
            if acc > best_acc :
                # perturb h only slightly
                # i.e., "reward" a higher accuracy
                k = 1
                best_acc = acc
                best_h = param_dict[h]
                param_dict[h] = best_h + p # perturb
                if printHelp : print('new best acc found:',best_acc,' ',h,' perturbed to',param_dict[h])
            else :
                # perturb h proportionally to the number of consecutive failed attempts
                # "punish" a lower accuracy
                param_dict[h] = best_h + p # perturb
                if printHelp : print(h,' resulted in a lower acc:',acc,' ',h,' perturbed to',param_dict[h])
                # the longer we go without finding a higher accuracy, the wider we want our
                # search to be. Thus, we scale our perturbations by a counter of consecutive failed points.
                k += 1
                
            # we do not want negative hyperparameter values,
            # nor do we want hyperparameter values outside of our range
            if param_dict[h] <= 0: param_dict[h] = 10
            elif param_dict[h] > hyper_parameter_range : param_dict[h] = hyper_parameter_range - 10
            
            if printHelp : print('......')
        if printHelp : print('best ',h,' value found was ',best_h)
        param_dict[h] = best_h
        if printHelp : print('/////////')
    
    best_point = def_point(param_dict)
    
    # convert best point to hyper parameters
    lr = point[0]/1000
    hs = int(round(point[1]))
    bs = int(round(point[2]))
    ne = int(round(point[3])/100)
    
    best_point = (lr, hs, bs, ne)
    
    # run best point found on test set
    #test_acc, experiment_number = runExperiment([best_point], 1, 'test', network_type, experiment_number)
    test_acc = faux_experiment(point)
    
    return test_acc, best_point, points_tested