import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Second Path
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(conv2_1)
    
    # Concatenate outputs
    concat_layer = Concatenate()([conv1_1, conv2_2])

    # 1x1 convolution for dimension alignment
    main_path_output = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(concat_layer)

    # Branch Path
    branch_path = input_layer

    # Add main path and branch path
    add_layer = Add()([main_path_output, branch_path])

    flatten_layer = Flatten()(add_layer)

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model