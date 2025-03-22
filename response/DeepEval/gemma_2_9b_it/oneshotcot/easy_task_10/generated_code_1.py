import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolution to increase dimensionality
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 depthwise separable convolution for feature extraction
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_multiplier=1)(x)

    # 1x1 convolution to reduce dimensionality
    x = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(x) 

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model