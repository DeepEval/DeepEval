import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Softmax

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # First block
    block1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    
    # Second block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)
    
    # Third block
    block3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)
    
    # Concatenate the outputs of the blocks
    concat_output = Concatenate()([block1, block2, block3])
    
    # Add a fully connected layer
    flatten = Flatten()(concat_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Return the constructed model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model