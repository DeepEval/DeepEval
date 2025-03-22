import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, BatchNormalization
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate the outputs of the main and branch paths
    combined_path = Add()([main_path, branch_path])
    
    # Flatten the result
    flatten_layer = Flatten()(combined_path)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model