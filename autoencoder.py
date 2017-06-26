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

    l_unpool_4 = lasagne.layers.Upscale2DLayer(l_deconv_3, scale_factor=2)

    l_deconv_4 = lasagne.layers.TransposedConv2DLayer(l_unpool_4, num_filters=3, filter_size=(3, 3), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.sigmoid)


    l_reconstruct = l_deconv_4

    return l_reconstruct



print("Loading data...")

flower= load_image.load_flower()
flower_train = flower[0:1300]


input_var = T.tensor4('inputs')

print("Building model and...")

net = build_ae(input_var)

sample_reconstruct = lasagne.layers.get_output(net)
loss = lasagne.objectives.squared_error(sample_reconstruct,input_var).mean()


# Get network params, with specifications of manually updated ones
params = lasagne.layers.get_all_params(net, trainable=True)
#updates = lasagne.updates.sgd(loss,params,learning_rate=0.01)
updates = lasagne.updates.adam(loss,params)

test_reconstruct = lasagne.layers.get_output(net,deterministic = True)
test_loss = lasagne.objectives.squared_error(test_reconstruct,input_var).mean()


# Compile theano function computing the training validation loss and accuracy:
train_fn = theano.function([input_var], loss, updates=updates)
#val_fn = theano.function([input_var], test_loss)


view_fn = theano.function([input_var], test_reconstruct)

# The training loop
print("Starting training...")
num_epochs = 20
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:

    train_err = 0
    train_batches = 0
    start_time = time.time()

    for batch in mini_batch.iterate_minibatches(flower_train, 100, shuffle=True):

        inputs= batch
        train_err += train_fn(inputs)
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

X_view = flower[1301:1350]
X_view = X_view[::7]
rec_img = view_fn(X_view)
for i in range(0,5):
    im = Image.fromarray(get_image_array(X_view,i),mode ="RGB")
    im.save('C:/Users/zjufe/PycharmProjects/Inpainting/out/View_{}.jpg'.format(i))
    im_c = Image.fromarray(get_image_array(rec_img,i))
    im_c.save('C:/Users/zjufe/PycharmProjects/Inpainting/out/Rec_{}.jpg'.format(i))
