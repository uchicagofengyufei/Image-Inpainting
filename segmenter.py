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
from scipy.misc import toimage


numpy.random.seed(25)
def build_ae(input_var):


    l_in = lasagne.layers.InputLayer(shape=(None,3,96,96),
                                     input_var=input_var)

    l_conv_1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(3, 3),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_conv_2 = lasagne.layers.Conv2DLayer(l_conv_1, num_filters=32, filter_size=(3, 3),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_pool_1 = lasagne.layers.MaxPool2DLayer(l_conv_2, pool_size=(2, 2), stride=2)

    l_conv_3 = lasagne.layers.Conv2DLayer(l_pool_1, num_filters=64, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(),
                                          b=None)
    l_conv_4 = lasagne.layers.Conv2DLayer(l_conv_3, num_filters=64, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(),
                                          b=None)
    l_pool_2 = lasagne.layers.MaxPool2DLayer(l_conv_4, pool_size=(2, 2), stride=2)


    l_conv_5 = lasagne.layers.Conv2DLayer(l_pool_2, num_filters=128, filter_size=(3, 3),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_pool_3 = lasagne.layers.MaxPool2DLayer(l_conv_5, pool_size=(2, 2), stride=2)

    l_conv_mid1 = lasagne.layers.Conv2DLayer(l_pool_3, num_filters=256, filter_size=(3, 3),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_conv_mid2 = lasagne.layers.Conv2DLayer(l_conv_mid1, num_filters=256, filter_size=(3, 3),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_unpool_1 = lasagne.layers.Upscale2DLayer(l_conv_mid2,scale_factor=2)

    l_deconv_r5 = lasagne.layers.TransposedConv2DLayer(l_unpool_1, num_filters=128, filter_size=(3, 3),crop = "same", stride=1,
                                                      nonlinearity=lasagne.nonlinearities.rectify)

    l_unpool_2 = lasagne.layers.Upscale2DLayer(l_deconv_r5, scale_factor=2)

    l_deconv_r4 = lasagne.layers.TransposedConv2DLayer(l_unpool_2, num_filters=64, filter_size=(3, 3), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.rectify)

    l_deconv_r3 = lasagne.layers.TransposedConv2DLayer(l_deconv_r4, num_filters=64, filter_size=(3, 3), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.rectify)

    l_unpool_3 = lasagne.layers.Upscale2DLayer(l_deconv_r3, scale_factor=2)

    l_deconv_r2 = lasagne.layers.TransposedConv2DLayer(l_unpool_3, num_filters=32, filter_size=(3, 3), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.rectify)

    l_deconv_r1 = lasagne.layers.TransposedConv2DLayer(l_deconv_r2, num_filters=1, filter_size=(3, 3), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.sigmoid)


    l_reconstruct = l_deconv_r1

    return l_reconstruct



print("Loading data...")

flower_train = load_image.load_seg_image(isTrain= True)
segment_train = load_image.load_segment().reshape(848,1,96,96)
print(segment_train.shape)
flower_test = load_image.load_seg_image(isTrain= False)

#load_image.show_image(flower_train,200)
#load_image.show_seg(segment_train[200],0)

input_var = T.tensor4('inputs')
target_var = T.tensor4('target')

print("Building model and...")

net = build_ae(input_var)

sample_reconstruct = lasagne.layers.get_output(net)
#loss = lasagne.objectives.squared_error(sample_reconstruct,target_var).mean()
loss = lasagne.objectives.binary_crossentropy(sample_reconstruct,target_var).mean()

# Get network params, with specifications of manually updated ones
params = lasagne.layers.get_all_params(net, trainable=True)
#updates = lasagne.updates.sgd(loss,params,learning_rate=0.01)
updates = lasagne.updates.adam(loss,params)

test_seg = lasagne.layers.get_output(net,deterministic = True)



# Compile theano function computing the training validation loss and accuracy:
train_fn = theano.function([input_var,target_var], loss, updates=updates)
#val_fn = theano.function([input_var], test_loss)

# The function to get generated picture, using target_var because I have to
# compute the loss on test set
view_fn = theano.function([input_var], test_seg)

# The training loop
print("Starting training...")
num_epochs = 50
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:

    train_err = 0
    train_batches = 0
    start_time = time.time()

    for batch in mini_batch.iterate_minibatches_pair(flower_train,segment_train, 100, shuffle=True):

        inputs,targets= batch
        train_err += train_fn(inputs,targets)
        train_batches += 1




    # Then we print the results for this epoch:

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))



    ################### View reconstruction###########


def get_image_array(X, index, shp=(96,96), channels=3):
    #print(X[index].shape)
    ret = (X[index] * 255.).transpose(1,2,0).astype(numpy.uint8)
    #print(ret.shape)
    return ret

test_seg = view_fn(flower_test)
for i in range(0,512):
    im = Image.fromarray(get_image_array(flower_test,i),mode ="RGB")
    im.save('C:/Users/zjufe/PycharmProjects/Inpainting/out/{}_View.jpg'.format(i))
    im_s = toimage(test_seg[i,0,:,:])
    im_s.save('C:/Users/zjufe/PycharmProjects/Inpainting/out/{}_Seg.jpg'.format(i))
