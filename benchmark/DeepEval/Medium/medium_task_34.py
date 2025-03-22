from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
        
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),  padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    dropped = Dropout(rate=0.3)(conv4)
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropped)

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    up5 = Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = Add()([conv3, up5])

    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up5)
    up6 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv6)
    up6 = Add()([conv2, up6])

    conv7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up6)
    up7 = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv7)
    up7 = Add()([conv1, up7])

    output_layer= Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(up7)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
