import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # First Block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch Path
    branch_path = input_layer

    # Combine main path and branch path
    block_output = Add()([conv2, branch_path])

    # Second Block
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block_output)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block_output)
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(block_output)
    output_block = Concatenate()([Flatten()(maxpool1), Flatten()(maxpool2), Flatten()(maxpool3)])

    # Final Layers
    bath_norm = BatchNormalization()(output_block)
    dense1 = Dense(units=128, activation='relu')(bath_norm)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model