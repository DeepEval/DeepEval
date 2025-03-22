import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Average Pooling paths
    def block1(input_tensor):
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)

        # Flatten each path
        path1_flat = Flatten()(path1)
        path2_flat = Flatten()(path2)
        path3_flat = Flatten()(path3)

        # Concatenate flattened paths
        output_tensor = Concatenate()([path1_flat, path2_flat, path3_flat])

        return output_tensor

    block1_output = block1(input_layer)
    
    # Fully connected layer between Block 1 and Block 2
    fc1 = Dense(units=256, activation='relu')(block1_output)

    # Reshape operation to convert output to 4D tensor for Block 2
    reshape_layer = Reshape((4, 4, 16))(fc1)  # Adjust dimensions as needed
    
    # Block 2: Feature extraction branches
    def block2(input_tensor):
        # Branch 1
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

        # Branch 2
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3
        branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)

        # Concatenate branches
        output_tensor = Concatenate()([Flatten()(branch1), Flatten()(branch2), Flatten()(branch3)])

        return output_tensor

    block2_output = block2(reshape_layer)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model