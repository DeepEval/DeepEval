import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    # Convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    # MaxPooling layer
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Block 1
    block1_output = block([maxpool1])
    # Batch normalization
    batchnorm1 = BatchNormalization()(block1_output)
    # Flatten layer
    flatten1 = Flatten()(batchnorm1)
    # Dense layer
    dense1 = Dense(units=128, activation='relu')(flatten1)
    
    # Branch 2
    # Convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
    # MaxPooling layer
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4)
    
    # Block 2
    block2_output = block([maxpool2])
    # Batch normalization
    batchnorm2 = BatchNormalization()(block2_output)
    # Flatten layer
    flatten2 = Flatten()(batchnorm2)
    # Dense layer
    dense2 = Dense(units=64, activation='relu')(flatten2)
    
    # Combine branches
    concat_layer = Concatenate(axis=-1)([dense1, dense2])
    
    # Additional dense layers
    dense3 = Dense(units=32, activation='relu')(concat_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

def block(input_tensor):
    # Three sequential convolutional layers
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # MaxPooling layer
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
    
    # Concatenate outputs
    output_tensor = Concatenate()([conv3, maxpool])
    
    return output_tensor