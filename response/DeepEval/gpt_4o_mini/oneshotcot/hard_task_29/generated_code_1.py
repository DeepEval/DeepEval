import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Concatenate, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Main path
    main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)

    # Branch path
    branch_path = input_layer

    # Combine both paths
    block1_output = Add()([main_path_conv2, branch_path])

    # Block 2
    pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)

    # Flatten the pooled outputs
    flatten1 = Flatten()(pooling1)
    flatten2 = Flatten()(pooling2)
    flatten3 = Flatten()(pooling3)

    # Concatenate the flattened outputs
    concatenated_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model