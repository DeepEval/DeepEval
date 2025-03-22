import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    block1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    block1 = Conv2D(32, (1, 1), activation='relu', padding='same')(block1)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)
    
    block2 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    block2 = Conv2D(64, (1, 1), activation='relu', padding='same')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2))(block2)

    # Branch path
    branch_conv = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Concatenate outputs from both paths
    concat = Concatenate()([block1, block2, branch_conv])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model