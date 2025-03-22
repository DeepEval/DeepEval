import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images (32x32 pixels with 3 color channels)
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    # Concatenate the input and output of the first layer
    concat1 = Concatenate()([input_layer, conv1])
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat1)
    # Concatenate the input and output of the second layer
    concat2 = Concatenate()([concat1, conv2])
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(concat2)
    # Concatenate the input and output of the third layer
    concat3 = Concatenate()([concat2, conv3])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concat3)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model