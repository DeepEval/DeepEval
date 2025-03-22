import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Conv2D, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Different scale max pooling and flatten
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    block1_output = Concatenate()([flat1, flat2, flat3])

    # Fully connected layer and reshape between Block 1 and Block 2
    fc1 = Dense(units=128, activation='relu')(block1_output)
    reshape = Reshape((4, 4, 8))(fc1)  # Reshaping to an appropriate 4D tensor for the next block

    # Block 2: Multiple branches with convolutions and pooling
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape)
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape)

    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Final classification layer
    flat_output = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flat_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model