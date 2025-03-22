import keras
from keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Path 2: Sequence of convolutions
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 7), activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(7, 1), activation='relu')(conv2_1)

    # Concatenate outputs from both paths
    concat_layer = keras.layers.Concatenate()([conv1_1, conv2_2])
    
    # 1x1 convolution to align dimensions
    main_path = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(concat_layer)
    
    # Branch path
    branch_path = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Add outputs from main and branch paths
    output = Add()([main_path, branch_path])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model