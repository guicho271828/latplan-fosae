#!/usr/bin/env python3

import config
import numpy as np
import numpy.random as random
import latplan
import latplan.model
from latplan.util        import curry
from latplan.util.tuning import *
from latplan.util.noise  import gaussian

import keras.backend as K
import tensorflow as tf

import os
import os.path

float_formatter = lambda x: "%.5f" % x
import sys
np.set_printoptions(formatter={'float_kind':float_formatter})

mode     = 'learn_dump'
sae_path = None

from keras.optimizers import Adam
from keras_adabound   import AdaBound
from keras_radam      import RAdam

import keras.optimizers

setattr(keras.optimizers,"radam", RAdam)
setattr(keras.optimizers,"adabound", AdaBound)

# default values
default_parameters = {
    'epoch'           : 1000,
    'batch_size'      : 1000,
    'optimizer'       : "radam",
    'max_temperature' : 5.0,
    'min_temperature' : 0.7,
    'N'               : None,
    'M'               : 2,
    'train_gumbel'    : True,    # if true, noise is added during training
    'train_softmax'   : True,    # if true, latent output is continuous
    'test_gumbel'     : False,   # if true, noise is added during testing
    'test_softmax'    : False,   # if true, latent output is continuous
    'dropout_z'       : False,
}
# hyperparameter tuning
parameters = {
    'beta'       :[-0.3,-0.1,0.0,0.1,0.3],
    'lr'         :[0.1,0.01,0.001],
    'U'          :[5,10,20],
    'A'          :[2,3,4],
    'P'          :[10,20,40,80,160,320],
    'layer'      :[100,400,1000],
    'dropout'    :[0.3,0.4,0.5],
    'noise'      :[0.1,0.2,0.4],
    'zerosuppress'       :[0.1,0.2,0.5],
    'zerosuppress_delay' :[0.05,0.1,0.2,0.3,0.5],
    'preencoder_dimention':[10,25,50,100,200,400],
    'preencoder_layers':[0,1,2],
    'preencoder_l1':[0.0, 0.00001, 0.0001, 0.001, 0.01],
    'preencoder_delay':[0.05,0.1,0.2,0.3,0.5],
    'preencoder_output_activation':[("relu","MSE"),("linear","MSE"),("sigmoid","MSE"),("sigmoid","BCE")],
    'loss':["BCE"],
    'eval':["MSE"],
}

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def plot_autoencoding_image(ae,test,train,plotmode):
    if 'plot' not in mode:
        return
    # rz = np.random.randint(0,2,(6,ae.parameters['N']))
    # ae.plot_autodecode(rz,ae.local("autodecoding_random.png"),verbose=True)

    test_plot = test[:6]
    train_plot = train[:6]
    
    from latplan.puzzles import shuffle_objects
    shuffled_test_plot = shuffle_objects(test[:6])
    shuffled_train_plot = shuffle_objects(train[:6])

    ae.plot(test_plot,ae.local("autoencoding_test.png"),verbose                     = True)
    ae.plot(train_plot,ae.local("autoencoding_train.png"),verbose                   = True)
    ae.plot(shuffled_test_plot,ae.local("autoencoding_test_shuffled.png"),verbose   = True)
    ae.plot(shuffled_train_plot,ae.local("autoencoding_train_shuffled.png"),verbose = True)

    ae.plot_render(test_plot,ae.local("render_test.png"),verbose                     = True,mode=plotmode)
    ae.plot_render(train_plot,ae.local("render_train.png"),verbose                   = True,mode=plotmode)
    ae.plot_render(shuffled_test_plot,ae.local("render_test_shuffled.png"),verbose   = True,mode=plotmode)
    ae.plot_render(shuffled_train_plot,ae.local("render_train_shuffled.png"),verbose = True,mode=plotmode)
    
    ae.plot_pos_neg(test[:30],ae.local("booleans_test.png"),verbose=True,mode=plotmode)
    ae.plot_pn_decisiontree(test,ae.local("test"),verbose=True,mode=plotmode)

def dump_all_actions(ae,configs,trans_fn,name="all_actions.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 5000
    loop = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                transitions = trans_fn(configs[begin:end])
                pre, suc = transitions[0], transitions[1]
                pre_b = ae.encode(pre,batch_size=1000).round().astype(int)
                suc_b = ae.encode(suc,batch_size=1000).round().astype(int)
                actions = np.concatenate((pre_b,suc_b), axis=1)
                np.savetxt(f,actions,"%d")

def dump_actions(ae,transitions,name="actions.csv",repeat=1):
    if 'dump' not in mode:
        return
    print(ae.local(name))
    pre, suc = transitions[0], transitions[1]
    if ae.parameters["test_gumbel"]:
        pre = np.repeat(pre,axis=0,repeats=10)
        suc = np.repeat(suc,axis=0,repeats=10)
    pre = ae.encode(pre,batch_size=1000)
    suc = ae.encode(suc,batch_size=1000)
    ae.dump_actions(pre,suc,batch_size=1000)

def dump_all_states(ae,configs,states_fn,name="all_states.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 5000
    loop = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                states = states_fn(configs[begin:end])
                states_b = ae.encode(states,batch_size=1000).round().astype(int)
                np.savetxt(f,states_b,"%d")

def dump_states(ae,states,name="states.csv",repeat=1):
    if 'dump' not in mode:
        return
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            np.savetxt(f,ae.encode(states,batch_size=1000).round().astype(int),"%d")

################################################################

# note: lightsout has epoch 200

def run(path,train,val,parameters,train_out=None,val_out=None,):
    if 'learn' in mode:
        if train_out is None:
            train_out = train
        if val_out is None:
            val_out = val
        ae, _, _ = simple_genetic_search(
            curry(nn_task, latplan.model.get(default_parameters["aeclass"]),
                  path,
                  train, train_out, val, val_out), # noise data is used for tuning metric
            default_parameters,
            parameters,
            path,
            limit=300,
            report_best= lambda net: net.save(),
        )
    elif 'reproduce' in mode:   # reproduce the best result from the grid search log
        if train_out is None:
            train_out = train
        if val_out is None:
            val_out = val
        ae, _, _ = reproduce(
            curry(nn_task, latplan.model.get(default_parameters["aeclass"]),
                  path,
                  train, train_out, val, val_out), # noise data is used for tuning metric
            default_parameters,
            parameters,
            path,
            report_best= lambda net: net.save(),
        )
        ae.save()
    else:
        ae = latplan.model.load(path)
    return ae

def show_summary(ae,train,test):
    if 'summary' in mode:
        ae.summary()
        ae.report(train, test_data=test, train_data_to=train, test_data_to=test)

################################################################

def puzzle(aeclass="FirstOrderAE",type='mnist',width=3,height=3,U=None,A=None,P=None,num_examples=6500,comment=None):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"] = aeclass

    # recording metadata
    default_parameters["activation"]   = "self.puzzle_activation"
    default_parameters["O"]            = 9 # object ID
    default_parameters["F"]            = 6 # object features
    parameters["preencoder_dimension"] = [0] # disable object embedding
    parameters["preencoder_layers"]    = [0.0] # disable object embedding
    parameters["preencoder_l1"]        = [0.0] # disable object embedding

    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["puzzle",type,width,height]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]

    configs = pre_configs
    objects = p.to_objects(configs, width, height, False)
    train = objects[:int(len(objects)*0.9)]
    val   = data[int(len(data)*0.9):int(len(data)*0.95)]
    test  = data[int(len(data)*0.95):]

    ae = run(os.path.join("samples",sae_path), train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae,test,train,"puzzle")

    dump_states (ae,objects)
    dump_actions(ae,p.object_transitions(width, height, configs=configs, one_per_state=True))
    dump_all_states (ae,all_configs,        lambda configs: p.to_objects(configs,width,height),)
    dump_all_actions(ae,all_configs,        lambda configs: p.object_transitions(width,height,configs),)

def bboxes_to_onehot(bboxes,X,Y):
    batch, objs = bboxes.shape[0:2]

    bboxes_grid = bboxes // 5
    x1 = bboxes_grid[:,:,0].flatten()
    y1 = bboxes_grid[:,:,1].flatten()
    x2 = bboxes_grid[:,:,2].flatten()
    y2 = bboxes_grid[:,:,3].flatten()
    x1o = np.eye(X)[x1].reshape((batch,objs,X))
    y1o = np.eye(Y)[y1].reshape((batch,objs,Y))
    x2o = np.eye(X)[x2].reshape((batch,objs,X))
    y2o = np.eye(Y)[y2].reshape((batch,objs,Y))
    bboxes_onehot = np.concatenate((x1o,y1o,x2o,y2o),axis=-1)
    del x1,y1,x2,y2,x1o,y1o,x2o,y2o
    return bboxes_onehot

def blocksworld(aeclass="FirstOrderAE",track="blocks-5-3",U=None,A=None,P=None,num_examples=6500,comment=None):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"] = aeclass

    with np.load(os.path.join(latplan.__path__[0],"puzzles",track+".npz")) as data:
        images = data['images'].astype(np.float32) / 256
        bboxes = data['bboxes']
        all_transitions_idx = data['transitions']

        picsize = data['picsize']
        patchsize = images.shape[2:]
        picsize_grid = (picsize // 5).astype(int)
        Y,X = picsize_grid[0], picsize_grid[1] # mind the axes!
        num_transitions = len(all_transitions_idx) // 2
        num_states, num_objs = bboxes.shape[0:2]
        num_examples = min(num_examples, num_transitions)
        print("loaded")

    default_parameters["picsize_grid"] = list(map(int,picsize_grid))
    default_parameters["picsize"]      = list(map(int,picsize))
    default_parameters["activation"]   = "self.blocks_activation"

    from latplan.puzzles.util import preprocess
    images = preprocess(images)
    bboxes_onehot = bboxes_to_onehot(bboxes,X,Y)
    all_states = np.concatenate((       images.reshape       ((num_states, num_objs,-1)),
                                        bboxes_onehot.reshape((num_states, num_objs,-1))),
                                axis=-1)
    del images, bboxes_onehot

    all_transitions_idx = all_transitions_idx.reshape((num_transitions, 2))
    random.shuffle(all_transitions_idx)
    transitions_idx = all_transitions_idx[:num_examples]
    transitions = all_states[transitions_idx.flatten()].reshape((num_examples, 2, num_objs, -1))
    states = transitions.reshape((num_examples*2, num_objs, -1))

    if num_states < 100:
        train = all_states
        val   = all_states
        test  = all_states
    else:
        train = states[:int(len(states)*0.9)]
        val   = states[int(len(states)*0.9):int(len(states)*0.95)]
        test  = states[int(len(states)*0.95):]

    print("checkpoint")

    ae = run(os.path.join(track,sae_path), train, val, parameters)
    show_summary(ae, train, test)

    plot_autoencoding_image(ae,test,train,"blocks")

    dump_states      (ae,states)
    dump_actions     (ae,transitions)
    dump_states      (ae,all_states,"all_states.csv")
    dump_all_actions (ae,all_transitions_idx,
                      lambda idx: all_states[idx.flatten()].reshape((len(idx),2,num_objs,-1)).transpose((1,0,2,3)))


def main():
    global mode, sae_path
    import sys
    if len(sys.argv) == 1:
        print({ k for k in dir(latplan.model)})
        gs = globals()
        print({ k for k in gs if hasattr(gs[k], '__call__')})
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)
        mode = sys.argv.pop(0)
        sae_path = "_".join(sys.argv)
        task = sys.argv.pop(0)

        def myeval(str):
            try:
                return eval(str)
            except:
                return str
        
        globals()[task](*map(myeval,sys.argv))
    
if __name__ == '__main__':
    try:
        main()
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()
