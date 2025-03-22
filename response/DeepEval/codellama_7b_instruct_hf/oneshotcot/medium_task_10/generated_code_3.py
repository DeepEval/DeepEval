import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)

    # Basic block
    def block(input_tensor):
        # Main path
        path1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = BatchNormalization()(path1)

        # Branch
        branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Addition
        output_tensor = Concatenate()([path1, branch])

        return output_tensor

    # First level
    level1 = block(input_tensor=max_pooling)

    # Second level
    level2 = block(input_tensor=level1)

    # Third level
    level3 = block(input_tensor=level2)

    # Global branch
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(level3)

    # Addition
    output_tensor = Concatenate()([level3, global_branch])

    # Average pooling
    output_tensor = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(output_tensor)

    # Flatten
    output_tensor = Flatten()(output_tensor)

    # Fully connected layers
    output_tensor = Dense(units=128, activation='relu')(output_tensor)
    output_tensor = Dense(units=64, activation='relu')(output_tensor)
    output_tensor = Dense(units=10, activation='softmax')(output_tensor)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_tensor)

    return model