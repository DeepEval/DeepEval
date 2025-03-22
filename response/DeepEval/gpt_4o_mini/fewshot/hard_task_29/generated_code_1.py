import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    
    # Branch path
    branch_path = input_layer

    # Combine both paths
    combined = Add()([main_conv2, branch_path])

    # Second block
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(combined)
    flatten1 = Flatten()(maxpool1)

    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(combined)
    flatten2 = Flatten()(maxpool2)

    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(combined)
    flatten3 = Flatten()(maxpool3)

    # Concatenate the results from the pooling layers
    output_block2 = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(output_block2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model