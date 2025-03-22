import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)
    
    # Second block
    block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(block2)
    block2 = Dropout(0.25)(block2)
    
    # Concatenate the outputs of the two blocks
    concat = Concatenate()([block1, block2])
    
    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()