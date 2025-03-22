import keras
from keras.layers import Input, Dense, Flatten, Concatenate, Lambda
from keras.models import Model
from keras.applications import VGG16

def dl_model():
    # Main path
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(64, (7, 7), padding='same')(input_layer)
    x = Lambda(lambda x: x / 64)(x)  # Layer normalization
    x = Conv2D(64, (1, 1), padding='same')(x)
    x = Conv2D(64, (1, 1), padding='same')(x)
    main_path = x
    
    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    branch_x = Conv2D(64, (1, 1), padding='same')(branch_input)
    branch_x = Conv2D(64, (1, 1), padding='same')(branch_x)
    branch_x = Conv2D(64, (1, 1), padding='same')(branch_x)
    branch_path = branch_x
    
    # Combine main and branch paths
    combined_path = Concatenate()([main_path, branch_path])
    
    # Flatten and classify
    flattened = Flatten()(combined_path)
    output = Dense(10, activation='softmax')(flattened)
    
    # Create and return model
    model = Model(inputs=[input_layer, branch_input], outputs=output)
    return model