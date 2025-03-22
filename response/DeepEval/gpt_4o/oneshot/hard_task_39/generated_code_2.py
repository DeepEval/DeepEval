import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, Conv2D, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Max Pooling with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    block1_output = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layer and reshape between Block 1 and Block 2
    fc_block1 = Dense(units=512, activation='relu')(block1_output)
    reshaped_block1 = Reshape(target_shape=(4, 4, 32))(fc_block1)

    # Block 2: Multiple branches for convolution and pooling
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_block1)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(reshaped_block1)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(reshaped_block1)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(reshaped_block1)

    block2_output = Concatenate()([path1, path2, path3, path4])

    # Flatten and final fully connected layer for classification
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model