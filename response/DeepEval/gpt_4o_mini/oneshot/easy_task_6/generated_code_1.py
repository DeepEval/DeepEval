import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)
    
    # Branch path
    branch_path = input_layer  # Direct connection from the input layer
    
    # Combine paths using addition
    combined = Add()([main_path_conv2, branch_path])
    
    # Flatten and Dense layer for classification
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model