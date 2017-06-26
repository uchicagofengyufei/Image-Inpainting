from __future__ import print_function
import sys
import time
import theano
import theano.tensor as T
import lasagne
import load_image
import mini_batch
import numpy
from PIL import Image


numpy.random.seed(25)
def build_ae(input_var):


    l_in = lasagne.layers.InputLayer(shape=(None,3,96,96),
                                     input_var=input_var)

    l_conv_1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(3, 3),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_pool_1 = lasagne.layers.MaxPool2DLayer(l_conv_1, pool_size=(2, 2), stride=2)

    l_conv_2 = lasagne.layers.Conv2DLayer(l_pool_1, num_filters=64, filter_size=(5, 5), pad='same',
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(),
                                          b=None)
    l_pool_2 = lasagne.layers.MaxPool2DLayer(l_conv_2, pool_size=(2, 2), stride=2)


    l_conv_3 = lasagne.layers.Conv2DLayer(l_pool_2, num_filters=128, filter_size=(5, 5),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_pool_3 = lasagne.layers.MaxPool2DLayer(l_conv_3, pool_size=(2, 2), stride=2)

    l_conv_4 = lasagne.layers.Conv2DLayer(l_pool_3, num_filters=128, filter_size=(5, 5),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_pool_4 = lasagne.layers.MaxPool2DLayer(l_conv_4, pool_size=(2, 2), stride=2)

    l_encode = lasagne.layers.Conv2DLayer(l_pool_4, num_filters=512, filter_size=(6, 6),pad = 'valid',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_decode = lasagne.layers.TransposedConv2DLayer(l_encode,num_filters=128,filter_size=(6,6),stride = 1,
                                                    nonlinearity=lasagne.nonlinearities.rectify)

    l_unpool_1 = lasagne.layers.Upscale2DLayer(l_decode,scale_factor=2)

    l_deconv_1 = lasagne.layers.TransposedConv2DLayer(l_unpool_1, num_filters=128, filter_size=(5, 5),crop = "same", stride=1,
                                                      nonlinearity=lasagne.nonlinearities.rectify)

    l_unpool_2 = lasagne.layers.Upscale2DLayer(l_deconv_1, scale_factor=2)

    l_deconv_2 = lasagne.layers.TransposedConv2DLayer(l_unpool_2, num_filters=64, filter_size=(5, 5), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.rectify)

    l_unpool_3 = lasagne.layers.Upscale2DLayer(l_deconv_2, scale_factor=2)

    l_deconv_3 = lasagne.layers.TransposedConv2DLayer(l_unpool_3, num_filters=32, filter_size=(5, 5), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.rectify)

    l_deconv_4 = lasagne.layers.TransposedConv2DLayer(l_deconv_3, num_filters=3, filter_size=(3, 3), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.sigmoid)


    l_reconstruct = l_deconv_4

    return l_reconstruct



print("Loading data...")

flower = load_image.load_flower()

flower_corrupt_train,flower_truth_train,test_corrupt,test_truth= load_image.load_flower_corrupted()
#load_image.show_image(flower_corrupt_train,3)


input_var = T.tensor4('inputs')
target_var = T.tensor4('target')

print("Building model and...")

net = build_ae(input_var)

sample_reconstruct = lasagne.layers.get_output(net)
loss = lasagne.objectives.squared_error(sample_reconstruct,target_var).mean()


# Get network params, with specifications of manually updated ones
params = lasagne.layers.get_all_params(net, trainable=True)
#updates = lasagne.updates.sgd(loss,params,learning_rate=0.01)
updates = lasagne.updates.adam(loss,params)

test_reconstruct = lasagne.layers.get_output(net,deterministic = True)
test_loss = lasagne.objectives.squared_error(test_reconstruct,target_var).mean()


# Compile theano function computing the training validation loss and accuracy:
train_fn = theano.function([input_var,target_var], loss, updates=updates)
#val_fn = theano.function([input_var], test_loss)

# The function to get generated picture, using target_var because I have to
# compute the loss on test set
view_fn = theano.function([input_var,target_var], [test_reconstruct,test_loss])

# The training loop
print("Starting training...")
num_epochs = 200
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:

    train_err = 0
    train_batches = 0
    start_time = time.time()

    for batch in mini_batch.iterate_minibatches_pair(flower_corrupt_train,flower_truth_train, 100, shuffle=True):

        inputs,targets= batch
        train_err += train_fn(inputs,targets)
        train_batches += 1




    # Then we print the results for this epoch:

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


    ################### View reconstruction###########


def get_image_array(X, index, shp=(96,96), channels=3):
    #print(X[index].shape)
    ret = (X[index] * 255.).transpose(1,2,0).astype(numpy.uint8)
    #print(ret.shape)
    return ret

test_ind = numpy.asarray(range(0,60))*23
X_original = flower[test_ind]
X_cor_view = test_corrupt
rec_img,tloss = view_fn(X_cor_view,test_truth)
print(tloss)
X_cor_view[:,:,25:73,25:73] = rec_img
#print(X_cor_view.shape)
for i in range(0,60):
    im = Image.fromarray(get_image_array(X_original,i),mode ="RGB")
    im.save('C:/Users/zjufe/PycharmProjects/Inpainting/out/View_{}.jpg'.format(i))
    im_c = Image.fromarray(get_image_array(X_cor_view,i))
    im_c.save('C:/Users/zjufe/PycharmProjects/Inpainting/out/Rec_{}.jpg'.format(i))

# train 0.005404
# test 0.046416