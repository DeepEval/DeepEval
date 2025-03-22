import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Dropout, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction through 3 pairs of <convolutional layer, max-pooling layer>
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Enhance generalization capabilities through <convolutional layer, Dropout layer, convolutional layer>
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool3)
    drop_out = Dropout(0.2)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop_out)

    # Upsampling through 3 pairs of <convolutional layer, transposed convolutional layer> with skip connections
    up6 = Conv2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv5))
    up6 = Concatenate()([conv3, up6])
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up6)
    up7 = Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv6))
    up7 = Concatenate()([conv2, up7])
    conv7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up7)
    up8 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv7))
    up8 = Concatenate()([conv1, up8])
    conv8 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up8)

    # 1x1 convolutional layer generates a probability output for 10 classes
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(conv8)

    # Reshape the output layer to (batch_size, 32, 32, 10)
    output_layer = Reshape((-1, 10))(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model