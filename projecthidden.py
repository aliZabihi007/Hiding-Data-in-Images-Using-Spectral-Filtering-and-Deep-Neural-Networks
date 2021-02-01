#import library
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import color
import cv2
import tensorflow as tf
from extra_keras_datasets import emnist
from copy import deepcopy

#add red filter to original image
#تابعی که در زیر مشاهده میکنیم  تصویر ها را دریافت کرده و بر روی آن فیلتر های قابل مشاهده تبدیل می کند .

def filtering_image(img):
    red_img = deepcopy(img)
    g_img = deepcopy(img)
    b_img = deepcopy(img)

    red_img[:, :, 1] = 0
    red_img[:, :, 2] = 0

    g_img[:, :, 0] = 0
    g_img[:, :, 2] = 0

    b_img[:, :, 0] = 0
    b_img[:, :, 1] = 0

    fig, ax = plt.subplots(ncols=2, nrows=2)
    filterimage = color.rgb2gray(red_img)
    # filterimage = cv2.cvtColor(filterimage, cv2.COLOR_BGR2GRAY)
    ax[0, 1].imshow(filterimage)
    ax[1, 0].imshow(g_img)
    ax[1, 1].imshow(b_img)
    plt.show()

    return filterimage

#بخش آموزش  شبکه با دیتا های از پیش داده شده و  تصاویر ما برای بخش پیبینی داده می شود 
#learning network with emnist

def learn(test_predict):
  #  object = tf.keras.datasets.mnist
    #(xtrain, ytrain), (xtest, ytest) = object.load_data()
    (xtrain, ytrain), (xtest, ytest) = emnist.load_data(type='digits')


    xtrain = xtrain / 255.0
    xtest = xtest / 255.0
 #   xtrain = np.concatenate(( [test_predict[3],test_predict[0]],xtrain))
  #  ytrain = np.concatenate(( [7,3],ytrain))
   

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28))
                                           , tf.keras.layers.Dense(256, activation='relu',name='hidden1'),
                                            tf.keras.layers.Dropout(0.2),
                                            tf.keras.layers.Dense(128, activation='relu',name='hidden2'),
                                         
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.summary()
    model.compile(optimizer='RMSprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    v=xtest[0:1000]
    yv=ytest[0:1000]
    history= model.fit(xtrain, ytrain, epochs=10,validation_data=(v, yv))
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1,10)
    plt.plot(loss_train, 'g')
    plt.plot( loss_val, 'b')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    
    plt.plot( loss_train, 'g')
    plt.plot(loss_val, 'b')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    predlist = list()
    print(model.evaluate(xtest, ytest))
    pred = model.predict(test_predict)
    print(pred.shape)
    print(test_predict.shape)

    for i in range(0,4,1):
        print(np.argmax(pred[i]))
        predlist.append(np.argmax(pred[i]))


#     print(np.argmax(pred[i]))
#    predlist.append(np.argmax(pred[i]))

    return predlist

# به نوعی پیش پردازشی است بر روی تصاویر صورت می پذیرد 
#preprocessing

def blackandwhite(img,teta):
    row, col = img.shape
    fil11 = np.array(np.zeros((row, col)))
    for i in range(row):
        for j in range(col):
            if (img[i][j] < teta):
                fil11[i][j] = 0
            else:
                fil11[i][j] = 200
    return fil11


if __name__ == '__main__':
    # img1 = io.imread("number1.png")
    img1 = io.imread('number1.png')
    img2 = io.imread("number2.png")
    img3 = io.imread("number3.png")
    img4 = io.imread("number4.png")

    fil1 = filtering_image(img1)
    fil1 = blackandwhite(fil1,0.202)
    fil2 = filtering_image(img2)
    fil2 = blackandwhite(fil2,0.207)
    fil3 = filtering_image(img3)
    fil3 = blackandwhite(fil3,0.17)
    fil4 = filtering_image(img4)
    fil4 = blackandwhite(fil4,0.209)

    filterimages = np.array([fil1,fil2,fil3,fil4])
    numbers= learn(filterimages)

    print(numbers)
    #/////////////////////////////////////////////////////////////////////
    img = io.imread("Untitled1.png")
    fil = filtering_image(img)
    io.imshow(fil)
    io.show()
    inputs=int(input(" لطفا پسورد را وارد کنید ؟\n"))
    sumadad=""
    for i in range(0,4,1):
        num=inputs^numbers[i]
        sumadad=sumadad+str(num)
    print(sumadad)
    if(sumadad=="0714"):
        print("پسورد امنیتی به درستی وارد شده . خوش امدید ")
    else:
        print("اوه. نه لطفا بار دیگر تلاش کنید ")


# imagedele = img2[:, :, 0]
# m = np.array([imagedele])
# print(m.shape)
