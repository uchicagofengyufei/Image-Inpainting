import numpy as np

def iterate_minibatches(inputs, batchsize, shuffle=False):

    #assert len(inputs) == len(targets)

    if shuffle:

        indices = np.arange(len(inputs))

        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):

        if shuffle:

            excerpt = indices[start_idx:start_idx + batchsize]

        else:

            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt]



def iterate_minibatches_pair(inputs,targets, batchsize, shuffle=False):

    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))

        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):

        if shuffle:

            excerpt = indices[start_idx:start_idx + batchsize]

        else:

            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt],targets[excerpt]
        
