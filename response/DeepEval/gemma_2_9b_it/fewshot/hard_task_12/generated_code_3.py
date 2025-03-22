import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 64))

    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv1)
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    main_path = Concatenate()([conv2_1, conv2_2])

    # Branch Path
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    branch_path = conv3

    # Combine Paths
    combined_path = keras.layers.add([main_path, branch_path])

    # Classification Layers
    flatten_layer = Flatten()(combined_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model