import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    block1_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    block1_output = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(block1_output)
    block1_output = Flatten()(block1_output)
    
    # Reshape the output for Block 2
    reshaped_output = Reshape((-1, 1))(block1_output)

    # Block 2
    block2_input = Input(shape=reshaped_output.shape)  # Input shape should match reshaped output
    block2_output = Conv2D(filters=32, kernel_size=(1x1), strides=(1, 1), padding='same', activation='relu')(block2_input)
    block2_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block2_output)
    block2_output = Flatten()(block2_output)

    # Branch processing
    branch1_output = Conv2D(filters=64, kernel_size=(1x1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_output = Conv2D(filters=64, kernel_size=(3x3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3_output = Conv2D(filters=64, kernel_size=(5x5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4_output = MaxPooling2D(pool_size=(3x3), strides=(1, 1), padding='valid')(input_layer)

    # Concatenate the outputs from all branches
    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])

    # Batch normalization and flattening
    concat_output = BatchNormalization()(concat_output)
    flattened_output = Flatten()(concat_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model