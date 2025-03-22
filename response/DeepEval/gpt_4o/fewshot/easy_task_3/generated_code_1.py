import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First <convolution, convolution, max pooling> block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)
    
    # Second <convolution, convolution, max pooling> block
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_2)

    # Third <convolution, convolution, convolution, max pooling> block
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3_1)
    conv3_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3_3)
    
    # Fourth <convolution, convolution, convolution, max pooling> block
    conv4_1 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(pool3)
    conv4_2 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4_1)
    conv4_3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv4_3)

    # Flatten and fully connected layers
    flatten = Flatten()(pool4)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model