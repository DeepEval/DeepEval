import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction through 3 pairs of convolutional layer and max-pooling layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)
    
    # Generalization capability enhancement through <convolutional layer, Dropout layer, convolutional layer>
    path1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    drop1 = Dropout(0.2)(path1)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop1)
    
    # Upsampling through 3 pairs of <convolutional layer, transposed convolutional layer> with skip connections
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    conv5 = concatenate([conv5, conv4])
    up1 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv5)

    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up1)
    conv6 = concatenate([conv6, conv2])
    up2 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv6)

    conv7 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up2)
    conv7 = concatenate([conv7, conv1])
    up3 = Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(conv7)
    
    # 1x1 convolutional layer to generate probability output for 10 classes
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(up3)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model