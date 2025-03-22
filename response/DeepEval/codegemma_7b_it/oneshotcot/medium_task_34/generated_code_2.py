import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Conv2DTranspose, Concatenate, Add
from tensorflow.keras import layers

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Feature extraction
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Feature enhancement
    conv4 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling3)
    drop4 = Dropout(0.5)(conv4)
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop4)

    # Feature upsampling
    up_sampling6 = Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv5)
    up_sampling6 = Concatenate()([up_sampling6, conv3])
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_sampling6)
    up_sampling7 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv6)
    up_sampling7 = Concatenate()([up_sampling7, conv2])
    conv7 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_sampling7)
    up_sampling8 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv7)
    up_sampling8 = Concatenate()([up_sampling8, conv1])
    conv8 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up_sampling8)

    # Output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv8)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model