import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    # First convolution and dropout block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.25)(main_path)
    
    # Second convolution and dropout block
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Dropout(0.25)(main_path)

    # Restore the number of channels
    main_path = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer  # Directly connect the input to the branch path

    # Combine both paths with an addition operation
    combined = Add()([main_path, branch_path])

    # Flattening layer
    flatten_layer = Flatten()(combined)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model