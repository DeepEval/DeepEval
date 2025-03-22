import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    # Main Path
    main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)

    # Branch Path
    branch_path = input_layer

    # Combine both paths with addition
    block1_output = Add()([main_path_conv2, branch_path])

    # Second Block
    # Max pooling layers with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(block1_output)

    # Flatten the pooled outputs
    flat_pool1 = Flatten()(pool1)
    flat_pool2 = Flatten()(pool2)
    flat_pool3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    concatenated_output = Concatenate()([flat_pool1, flat_pool2, flat_pool3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
# model = dl_model()
# model.summary()