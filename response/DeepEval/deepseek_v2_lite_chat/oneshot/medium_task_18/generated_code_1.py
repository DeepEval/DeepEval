import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First set of convolutional and pooling operations
    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    
    # Second set of convolutional and pooling operations
    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    
    # Third set of convolutional and pooling operations
    conv_3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    
    # Fourth set of convolutional and pooling operations
    conv_4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(pool_3)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

    # Concatenate the outputs from all sets of operations
    concat = Concatenate()([pool_4, conv_4, pool_3, conv_3, pool_2, conv_2, pool_1, conv_1])
    
    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()