import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First Block: Main Path and Branch Path with Addition
    # Main Path
    main_conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)

    # Branch Path (Direct Connection to Input)
    branch_path = input_layer

    # Combining Main Path and Branch Path
    block1_output = Add()([main_conv2, branch_path])

    # Second Block: Multiple Max Pooling Layers with Concatenation
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_output)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(block1_output)

    # Flatten and Concatenate
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    block2_output = Concatenate()([flat1, flat2, flat3])

    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(block2_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model