import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Conv2D, Reshape
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    # Path 1: MaxPooling (1x1)
    path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path1 = Flatten()(path1)
    path1 = Dropout(0.5)(path1)

    # Path 2: MaxPooling (2x2)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path2 = Flatten()(path2)
    path2 = Dropout(0.5)(path2)

    # Path 3: MaxPooling (4x4)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    path3 = Flatten()(path3)
    path3 = Dropout(0.5)(path3)

    # Concatenate outputs of all paths
    block1_output = Concatenate()([path1, path2, path3])

    # Fully connected layer
    fc1 = Dense(128, activation='relu')(block1_output)
    
    # Reshape to a 4D tensor suitable for Block 2
    reshaped_output = Reshape((1, 1, 128))(fc1)  # This is an example; adjust shape based on needs.

    # Block 2
    # Path 1: 1x1 Conv
    path1_block2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)

    # Path 2: 1x1 Conv followed by 1x7 and 7x1 Conv
    path2_block2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path2_block2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path2_block2)
    path2_block2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path2_block2)

    # Path 3: 1x1 Conv followed by alternating 7x1 and 1x7 Conv
    path3_block2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(reshaped_output)
    path3_block2 = Conv2D(filters=32, kernel_size=(7, 1), padding='same', activation='relu')(path3_block2)
    path3_block2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(path3_block2)

    # Path 4: Average Pooling with a 1x1 Conv
    path4_block2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped_output)
    path4_block2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4_block2)

    # Concatenate outputs of all paths in Block 2
    block2_output = Concatenate()([path1_block2, path2_block2, path3_block2, path4_block2])

    # Flatten the concatenated output from Block 2
    flatten_block2 = Flatten()(block2_output)

    # Fully connected layers for final classification
    dense1 = Dense(64, activation='relu')(flatten_block2)
    output_layer = Dense(10, activation='softmax')(dense1)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model