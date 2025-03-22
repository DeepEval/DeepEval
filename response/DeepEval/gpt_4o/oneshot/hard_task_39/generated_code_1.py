import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Reshape, Conv2D, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Max Pooling with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten and concatenate results of Block 1
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    block1_output = Concatenate()([flat1, flat2, flat3])

    # Between Block 1 and Block 2: Fully connected layer and reshape
    fc = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape((4, 4, 8))(fc)  # Reshape to 4D tensor for Block 2

    # Block 2: Multiple branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Final classification layers
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model