import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the first special block
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block1)
    block1 = Flatten()(block1)
    block1 = Dropout(rate=0.5)(block1)
    
    # Define the second special block
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    block2 = Flatten()(block2)
    block2 = Dropout(rate=0.5)(block2)
    
    # Concatenate the outputs of the two special blocks
    concatenated = Concatenate()([block1, block2])
    
    # Apply batch normalization and flatten the result
    batch_normalized = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_normalized)
    
    # Define the fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model