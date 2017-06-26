from PIL import Image
import numpy
from scipy.misc import toimage



def load_flower():
    flower  =numpy.load("flowers.npy")
    flower  = 1.0*flower / 256
    #print(numpy.max(flower))
    return flower.astype(numpy.float32)

def load_flower_corrupted():
    test_ind = numpy.asarray(range(0, 60)) * 23
    flower = numpy.load("flowers.npy")
    flower2 = numpy.load("flowers.npy")
    flower = 1.0 * flower / 255
    flower2 = 1.0 * flower2 / 255

    flower_test = flower[test_ind].astype(numpy.float32)
    flower_train = numpy.delete(flower,test_ind,axis = 0).astype(numpy.float32)
    flower_test[:, :, 25:73, 25:73] = 0
    flower_train[:, :, 25:73, 25:73] = 0

    flower_truth_test = flower2[test_ind].astype(numpy.float32)
    flower_truth_train = numpy.delete(flower2, test_ind, axis=0).astype(numpy.float32)
    flower_truth_test = flower_truth_test[:,:,25:73,25:73]
    flower_truth_train = flower_truth_train[:,:,25:73,25:73]


    return flower_train,flower_truth_train,flower_test,flower_truth_test



def load_flower_corrupted_augment():
    test_ind = numpy.asarray(range(0, 60)) * 23+1
    flower = numpy.load("flowers.npy")
    flower2 = numpy.load("flowers.npy")
    flower = 1.0 * flower / 255
    flower2 = 1.0 * flower2 / 255

    flower_test = flower[test_ind].astype(numpy.float32)
    flower_train = numpy.delete(flower,test_ind,axis = 0).astype(numpy.float32)
    flower_test[:, :, 25:73, 25:73] = 0
    flower_train[:, :, 25:73, 25:73] = 0

    flower_truth_test = flower2[test_ind].astype(numpy.float32)
    flower_truth_train = numpy.delete(flower2, test_ind, axis=0).astype(numpy.float32)
    flower_truth_test = flower_truth_test[:,:,17:81,17:81]
    flower_truth_train = flower_truth_train[:,:,17:81,17:81]


    return flower_train[:,:,17:81,17:81],flower_truth_train,flower_test[:,:,17:81,17:81],flower_truth_test


def show_image(img_set,index):
    img_arr = img_set[index]
    img_arr = img_arr.transpose(1, 2, 0)*255
    img_arr = img_arr.astype(numpy.uint8)

    img = Image.fromarray(img_arr)
    img.show()



def load_flower_corrupted_center():
    test_ind = numpy.asarray(range(0, 60)) * 23+2
    flower = numpy.load("flowers.npy")
    flower2 = numpy.load("flowers.npy")
    flower = 1.0 * flower / 255
    flower2 = 1.0 * flower2 / 255

    flower_test = flower[test_ind].astype(numpy.float32)
    flower_train = numpy.delete(flower,test_ind,axis = 0).astype(numpy.float32)
    flower_test[:, :, 31:67, 31:67] = 0
    flower_train[:, :, 31:67, 31:67] = 0

    flower_truth_test = flower2[test_ind].astype(numpy.float32)
    flower_truth_train = numpy.delete(flower2, test_ind, axis=0).astype(numpy.float32)
    flower_truth_test = flower_truth_test[:,:,31:67,31:67]
    flower_truth_train = flower_truth_train[:,:,31:67,31:67]

    return flower_train,flower_truth_train,flower_test,flower_truth_test



def load_flower_corrupted_leftup():
    test_ind = numpy.asarray(range(0, 60)) * 23+2
    flower = numpy.load("flowers.npy")
    flower2 = numpy.load("flowers.npy")
    flower = 1.0 * flower / 255
    flower2 = 1.0 * flower2 / 255

    flower_test = flower[test_ind].astype(numpy.float32)
    flower_train = numpy.delete(flower,test_ind,axis = 0).astype(numpy.float32)
    flower_test[:, :, 13:49, 13:49] = 0
    flower_train[:, :, 13:49, 13:49] = 0

    flower_truth_test = flower2[test_ind].astype(numpy.float32)
    flower_truth_train = numpy.delete(flower2, test_ind, axis=0).astype(numpy.float32)
    flower_truth_test = flower_truth_test[:,:,13:49,13:49]
    flower_truth_train = flower_truth_train[:,:,13:49,13:49]

    return flower_train,flower_truth_train,flower_test,flower_truth_test


def load_flower_corrupted_leftdown():
    test_ind = numpy.asarray(range(0, 60)) * 23 + 2
    flower = numpy.load("flowers.npy")
    flower2 = numpy.load("flowers.npy")
    flower = 1.0 * flower / 255
    flower2 = 1.0 * flower2 / 255

    flower_test = flower[test_ind].astype(numpy.float32)
    flower_train = numpy.delete(flower, test_ind, axis=0).astype(numpy.float32)
    flower_test[:, :, 49:85, 13:49] = 0
    flower_train[:, :, 49:85, 13:49] = 0

    flower_truth_test = flower2[test_ind].astype(numpy.float32)
    flower_truth_train = numpy.delete(flower2, test_ind, axis=0).astype(numpy.float32)
    flower_truth_test = flower_truth_test[:, :, 49:85, 13:49]
    flower_truth_train = flower_truth_train[:, :, 49:85, 13:49]

    return flower_train, flower_truth_train, flower_test, flower_truth_test


def load_flower_corrupted_rightup():
    test_ind = numpy.asarray(range(0, 60)) * 23 + 2
    flower = numpy.load("flowers.npy")
    flower2 = numpy.load("flowers.npy")
    flower = 1.0 * flower / 255
    flower2 = 1.0 * flower2 / 255

    flower_test = flower[test_ind].astype(numpy.float32)
    flower_train = numpy.delete(flower, test_ind, axis=0).astype(numpy.float32)
    flower_test[:, :, 13:49, 49:85] = 0
    flower_train[:, :, 13:49, 49:85] = 0

    flower_truth_test = flower2[test_ind].astype(numpy.float32)
    flower_truth_train = numpy.delete(flower2, test_ind, axis=0).astype(numpy.float32)
    flower_truth_test = flower_truth_test[:, :, 13:49, 49:85]
    flower_truth_train = flower_truth_train[:, :, 13:49, 49:85]

    return flower_train, flower_truth_train, flower_test, flower_truth_test


def load_flower_corrupted_rightdown():
    test_ind = numpy.asarray(range(0, 60)) * 23 + 2
    flower = numpy.load("flowers.npy")
    flower2 = numpy.load("flowers.npy")
    flower = 1.0 * flower / 255
    flower2 = 1.0 * flower2 / 255

    flower_test = flower[test_ind].astype(numpy.float32)
    flower_train = numpy.delete(flower, test_ind, axis=0).astype(numpy.float32)
    flower_test[:, :, 49:85, 49:85] = 0
    flower_train[:, :, 49:85, 49:85] = 0

    flower_truth_test = flower2[test_ind].astype(numpy.float32)
    flower_truth_train = numpy.delete(flower2, test_ind, axis=0).astype(numpy.float32)
    flower_truth_test = flower_truth_test[:, :, 49:85, 49:85]
    flower_truth_train = flower_truth_train[:, :, 49:85, 49:85]

    return flower_train, flower_truth_train, flower_test, flower_truth_test



def load_flower_random_mask():
    A1, B1, C1, D1 = load_flower_corrupted_center()
    A2, B2, C2, D2 = load_flower_corrupted_leftup()
    A3, B3, C3, D3 = load_flower_corrupted_leftdown()
    A4, B4, C4, D4 = load_flower_corrupted_rightup()
    A5, B5, C5, D5 = load_flower_corrupted_rightdown()

    flower_train = numpy.concatenate((A1,A2,A3,A4,A5),axis = 0)
    flower_truth_train = numpy.concatenate((B1,B2,B3,B4,B5),axis = 0)
    flower_test = numpy.concatenate((C1, C2, C3, C4, C5), axis=0)
    flower_truth_test = numpy.concatenate((D1, D2, D3, D4, D5), axis=0)

    return flower_train, flower_truth_train, flower_test, flower_truth_test

def get_mask_type():
    type = numpy.zeros(60*5)
    type[0:60] = 0
    type[61:120] = 1
    type[121:180] = 2
    type[181:240] = 3
    type[241:300] = 4
    return type

#print(get_mask_type())


def reconstruct(test_img,gen_img):
    test_img[0:60, :, 31:67, 31:67] = gen_img[0:60]
    test_img[60:120, :, 13:49, 13:49] = gen_img[60:120]
    test_img[120:180, :, 49:85, 13:49] = gen_img[120:180]
    test_img[180:240, :, 13:49, 49:85] = gen_img[180:240]
    test_img[240:300, :, 49:85, 49:85] = gen_img[240:300]
    return test_img


def load_segment():
    seg_data = numpy.load("flowers_seg.npy")
    seg_data[ seg_data != 1 ] = 0
    return seg_data.astype(numpy.float32)

def load_seg_image(isTrain):
    flower = numpy.load("flowers.npy")
    seg_ind = numpy.load("seg_ind.npy")
    if isTrain:
        flower = flower[seg_ind]
        print("Load Training set flower")
    else:

        flower = numpy.delete(flower, seg_ind, axis=0)
        print("Load Test set flower")

    print(flower.shape)
    flower = 1.0 * flower / 256
    # print(numpy.max(flower))
    return flower.astype(numpy.float32)

def show_seg(seg_set,index):
    img = seg_set[index]
    img2 = img
    img2[img2 != 1] = 0
    toimage(img2 * 0.5).show()


#A,B,C,D = load_flower_random_mask()

#flower = load_flower_corrupted()
#show_image(A,5600)
#show_image(B,5600)
#A = load_seg_image()
#B = load_segment()
#show_image(A,106)
#show_seg(B,106)


"""
# Read  Image from jpg and save
flower = numpy.zeros([1360,3,96,96])
for i in range(1,1361):
    if(i<10):
        name = "000{}".format(i)
    elif(i>=10 and i<100):
        name = "00{}".format(i)
    elif(i>=100 and i<1000):
        name = "0{}".format(i)
    else:
       name = "{}".format(i)

    img = Image.open("C:/Users/zjufe/PycharmProjects/Inpainting/jpg/"
                    +"image_"+name+".jpg")
    #img.show()
    img = img.crop((96,96,480,480))
    img = img.resize((96,96),Image.LANCZOS)
    #img.show()
    img_arr = numpy.asarray(img).transpose(2,0,1)
    flower[i-1] = img_arr

numpy.save(file="flowers.npy",arr = flower)

"""
"""
flower = numpy.load("flowers.npy")
print(flower.shape)
print(numpy.max(flower))
img_arr = flower[0]
img_arr = img_arr.transpose(1,2,0)
img_arr = img_arr.astype(numpy.uint8)

img = Image.fromarray(img_arr)

img.show()
"""
