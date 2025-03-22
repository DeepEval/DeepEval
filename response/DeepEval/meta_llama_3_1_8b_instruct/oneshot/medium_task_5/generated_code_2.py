import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_block1)

    conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_block2)

    # Branch path
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_branch)
    max_pooling_branch = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_branch)

    # Combine the outputs of both paths
    combined_output = Add()([max_pooling, max_pooling_branch])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bath_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model