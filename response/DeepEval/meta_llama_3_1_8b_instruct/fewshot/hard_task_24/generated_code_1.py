import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1: extract local features through a 3x3 convolutional layer
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch 2: sequentially pass through a max pooling layer, a 3x3 convolutional layer, and then an upsampling layer
    pool2_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2_2)
    upsample2_2 = UpSampling2D(size=(2, 2))(conv2_2)
    
    # Branch 3: sequentially pass through a max pooling layer, a 3x3 convolutional layer, and then an upsampling layer
    pool3_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv3_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3_3)
    upsample3_3 = UpSampling2D(size=(2, 2))(conv3_3)
    
    # Fuse the outputs of all branches through concatenation and pass through another 1x1 convolutional layer
    concat = Concatenate()([conv2_1, upsample2_2, upsample3_3])
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Flatten the output and pass it through three fully connected layers to produce a 10-class classification result
    flatten = Flatten()(conv4)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model