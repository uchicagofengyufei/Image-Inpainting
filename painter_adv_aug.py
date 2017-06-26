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


    l_conv_4 = lasagne.layers.Conv2DLayer(l_pool_3, num_filters=256, filter_size=(3, 3),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)


    l_encode = lasagne.layers.Conv2DLayer(l_conv_4, num_filters=256, filter_size=(3, 3),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),b = None)

    l_unpool_1 = lasagne.layers.Upscale2DLayer(l_encode,scale_factor=2)

    l_deconv_1 = lasagne.layers.TransposedConv2DLayer(l_unpool_1, num_filters=128, filter_size=(5, 5),crop = "same", stride=1,
                                                      nonlinearity=lasagne.nonlinearities.rectify)

    l_unpool_2 = lasagne.layers.Upscale2DLayer(l_deconv_1, scale_factor=2)

    l_deconv_2 = lasagne.layers.TransposedConv2DLayer(l_unpool_2, num_filters=64, filter_size=(5, 5), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.rectify)


    l_deconv_3 = lasagne.layers.TransposedConv2DLayer(l_deconv_2, num_filters=32, filter_size=(5, 5), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.rectify)

    l_deconv_4 = lasagne.layers.TransposedConv2DLayer(l_deconv_3, num_filters=3, filter_size=(3, 3), crop="same",
                                                      stride=1,nonlinearity=lasagne.nonlinearities.sigmoid)


    l_reconstruct = l_deconv_4

    return l_reconstruct


def build_discriminator(input_pic,surrounding):

    l_in1 = lasagne.layers.InputLayer(shape=(None, 3, 48, 48),
                                     input_var=input_pic)

    l_pad_1 = lasagne.layers.PadLayer(l_in1,width = 24)

    l_in2 = lasagne.layers.InputLayer(shape=(None, 3, 96, 96),
                                     input_var=surrounding)

    l_in = lasagne.layers.ElemwiseSumLayer((l_pad_1,l_in2))

    l_conv_1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(),
                                          b=None)

    l_pool_0 = lasagne.layers.MaxPool2DLayer(l_conv_1, pool_size=(2, 2), stride=2) #48

    l_conv_2 = lasagne.layers.Conv2DLayer(l_pool_0, num_filters=32, filter_size=(3, 3), pad='same',
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(),
                                          b=None)

    l_pool_1 = lasagne.layers.MaxPool2DLayer(l_conv_2, pool_size=(2, 2), stride=2)#24
    #l_pool_1 = lasagne.layers.Pool2DLayer(l_conv_2, pool_size=(2, 2), stride=2,mode = 'average_inc_pad')

    l_conv_3 = lasagne.layers.Conv2DLayer(l_pool_1, num_filters=32, filter_size=(5, 5), pad='same',
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(),
                                          b=None)


    l_pool_2 = lasagne.layers.MaxPool2DLayer(l_conv_3, pool_size=(2, 2), stride=2)#12
    #l_pool_2 = lasagne.layers.Pool2DLayer(l_conv_4, pool_size=(2, 2), stride=2,mode = 'average_inc_pad')

    l_conv_5 = lasagne.layers.Conv2DLayer(l_pool_2, num_filters=32, filter_size=(7, 7), pad='same',
                                          nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(),
                                          b=None)

    l_pool_3 = lasagne.layers.MaxPool2DLayer(l_conv_5, pool_size=(2, 2), stride=2)#6
    #l_pool_3 = lasagne.layers.Pool2DLayer(l_conv_5, pool_size=(2, 2), stride=2,mode = 'average_inc_pad')


    l_hid_1 = lasagne.layers.DenseLayer(
        l_pool_3, num_units=128,
        nonlinearity=lasagne.nonlinearities.rectify, b=None)

    l_out = lasagne.layers.DenseLayer(
        l_hid_1, num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax, b=None)

    return l_out


print("Loading data...")

flower = load_image.load_flower()

flower_corrupt_train,flower_truth_train,test_corrupt,test_truth= load_image.load_flower_corrupted()
#load_image.show_image(flower_corrupt_train,3)


input_var = T.tensor4('inputs')
target_var = T.tensor4('target')
adv_input = T.tensor4('adv_input')
adv_target = T.ivector('adv_target')

print("Building model and...")

# Build the Generator
net = build_ae(input_var)
sample_reconstruct = lasagne.layers.get_output(net)
loss = lasagne.objectives.squared_error(sample_reconstruct,target_var).mean()

# Get network params, with specifications of manually updated ones
params = lasagne.layers.get_all_params(net, trainable=True)
updates = lasagne.updates.sgd(loss,params,learning_rate=0.01)
#updates = lasagne.updates.adam(loss,params)



# Building the adversarial distriminator network
adv_net = build_discriminator(sample_reconstruct,input_var)
adv_predict = lasagne.layers.get_output(adv_net)
adv_loss = lasagne.objectives.categorical_crossentropy(adv_predict,adv_target).mean()

adv_params = lasagne.layers.get_all_params(adv_net, trainable=True)
#adv_updates = lasagne.updates.sgd(adv_loss,adv_params,learning_rate=0.01)
adv_updates = lasagne.updates.adam(adv_loss,adv_params)

test_reconstruct = lasagne.layers.get_output(net,deterministic = True)
test_loss = lasagne.objectives.squared_error(test_reconstruct,target_var).mean()


#combined_loss = T.log10(loss) - 0.01*adv_loss
combined_loss = 0.99*loss - 0.01*adv_loss
combined_update = lasagne.updates.adam(combined_loss,params)


# Compile theano function computing the training validation loss and accuracy:
train_fn = theano.function([input_var,target_var], loss, updates=updates)
generator_fn = theano.function([input_var,target_var,adv_target],[combined_loss,sample_reconstruct],updates = combined_update)
discriminator_fn = theano.function([sample_reconstruct,input_var,adv_target],adv_loss,updates = adv_updates)
#val_fn = theano.function([input_var], test_loss)


view_fn = theano.function([input_var,target_var], [test_reconstruct,test_loss])

# The training loop
print("Starting training...")
num_epochs = 120
gen_label = numpy.zeros(100).astype(numpy.int32)
adv_label = numpy.zeros(200).astype(numpy.int32)
adv_label[100:200] = 1
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:

    train_err = 0
    adv_err = 0
    train_batches = 0
    start_time = time.time()

    for batch in mini_batch.iterate_minibatches_pair(flower_corrupt_train,flower_truth_train, 100, shuffle=True):
        if epoch<50:
            inputs,targets= batch
            train_err += train_fn(inputs,targets)
            train_batches += 1
        else:
            inputs, targets = batch
            err,gen_pic = generator_fn(inputs,targets,gen_label)
            train_err += err

            adv_feed_center = numpy.concatenate((gen_pic,targets),axis = 0)
            adv_feed_surrounding = numpy.concatenate((inputs,inputs),axis = 0)


            adv_err += discriminator_fn(adv_feed_center,adv_feed_surrounding,adv_label)

            train_batches += 1






    # Then we print the results for this epoch:

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  discriminator loss:\t\t{:.6f}".format(adv_err / train_batches))
    #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


    ################### View reconstruction###########


def get_image_array(X, index, shp=(96,96), channels=3):
    #print(X[index].shape)
    ret = (X[index] * 255.).transpose(1,2,0).astype(numpy.uint8)
    #print(ret.shape)
    return ret

test_ind = numpy.asarray(range(0,60))*23+1
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
