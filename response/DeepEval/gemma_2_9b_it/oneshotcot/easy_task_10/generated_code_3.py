import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 Convolutional Layer
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), activation='relu')(input_layer)
    
    # 3x3 Depthwise Separable Convolutional Layer
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu', depthwise=True)(x)
    
    # 1x1 Convolutional Layer
    x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)

    # Flatten and Fully Connected Layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model