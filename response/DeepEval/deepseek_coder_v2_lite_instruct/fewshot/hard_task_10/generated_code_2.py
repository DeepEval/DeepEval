import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv_sequence = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(conv_1x1)
    conv_sequence = Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(conv_sequence)
    
    # Branch path
    branch = input_layer
    
    # Merge main path and branch
    added = Add()([conv_sequence, branch])
    
    # 1x1 convolution to align dimensions
    final_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(added)
    
    # Flatten the output
    flatten_layer = Flatten()(final_path)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model