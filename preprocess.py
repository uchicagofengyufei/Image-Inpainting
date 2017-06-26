from PIL import Image
import numpy



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

    img = Image.open("C:/Users/zjufe/PycharmProjects/Inpainting/seg/"
                    +"image_"+name+".jpg")
    #img.show()
    img = img.crop((96,96,480,480))
    img = img.resize((96,96),Image.LANCZOS)
    #img.show()
    img_arr = numpy.asarray(img).transpose(2,0,1)
    flower[i-1] = img_arr

numpy.save(file="flowers.npy",arr = flower)