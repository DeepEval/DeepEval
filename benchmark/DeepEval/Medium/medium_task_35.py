from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
        
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    dropped = Dropout(rate=0.3)(conv3)

    conv3 = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropped)
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = Add()([conv2, up4])

    conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),padding='same', activation='relu')(up4)
    up5 = UpSampling2D(size=(2, 2))(conv5)
    up5 = Add()([conv1, up5])

    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(up5)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
