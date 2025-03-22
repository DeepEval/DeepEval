import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the input layer
    input_layer = Input(shape=input_shape)
    
    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Dropout(0.2)(branch1)  # Apply dropout to mitigate overfitting
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(0.2)(branch2)  # Apply dropout to mitigate overfitting
    
    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Dropout(0.2)(branch3)  # Apply dropout to mitigate overfitting
    
    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    branch4 = Dropout(0.2)(branch4)  # Apply dropout to mitigate overfitting
    
    # Concatenate the outputs from all branches
    concatenated_output = Concatenate()([branch1, branch2, branch3, branch4])
    
    # Apply batch normalization
    batch_norm = BatchNormalization()(concatenated_output)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Apply three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model