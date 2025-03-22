from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Multi-Scale Max Pooling
    pool1_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool1_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool1_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    # Flattening pooling results
    flat1_1x1 = Flatten()(pool1_1x1)
    flat1_2x2 = Flatten()(pool1_2x2)
    flat1_4x4 = Flatten()(pool1_4x4)
    
    # Concatenating flattened vectors
    concat_block1 = Concatenate()([flat1_1x1, flat1_2x2, flat1_4x4])
    
    # Fully connected layer between Block 1 and Block 2
    fc_block1 = Dense(units=128, activation='relu')(concat_block1)
    
    # Reshape operation to convert to 4D tensor for Block 2
    reshape_block1 = Reshape((4, 4, 8))(fc_block1)  # Assuming a reshape to 4x4x8

    # Block 2: Multi-Branch Convolutions and Max Pooling
    conv_1x1 = Conv2D(8, (1, 1), activation='relu', padding='same')(reshape_block1)
    conv_3x3 = Conv2D(8, (3, 3), activation='relu', padding='same')(reshape_block1)
    conv_5x5 = Conv2D(8, (5, 5), activation='relu', padding='same')(reshape_block1)
    pool_3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshape_block1)

    # Concatenating outputs from all branches
    concat_block2 = Concatenate()([conv_1x1, conv_3x3, conv_5x5, pool_3x3])

    # Flatten and Fully Connected Layer for Classification
    flatten_output = Flatten()(concat_block2)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # 10 classes for MNIST

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model