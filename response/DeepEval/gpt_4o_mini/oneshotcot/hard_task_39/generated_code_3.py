import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Concatenate, Reshape, Conv2D
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Three max pooling layers with varying scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    # Flatten the pooling results
    flat_pool1 = Flatten()(pool1)
    flat_pool2 = Flatten()(pool2)
    flat_pool3 = Flatten()(pool3)

    # Concatenate the flattened results
    block1_output = Concatenate()([flat_pool1, flat_pool2, flat_pool3])

    # Fully connected layer and reshape operation
    dense1 = Dense(units=64, activation='relu')(block1_output)
    reshaped_output = Reshape((4, 4, 4))(dense1)  # Assuming we want to reshape it to (4, 4, 4)

    # Block 2: Multiple branches
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshaped_output)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped_output)

    # Concatenate outputs from all branches
    block2_output = Concatenate()([path1, path2, path3, path4])

    # Final flattening and dense layer for classification
    final_flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(final_flatten)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model