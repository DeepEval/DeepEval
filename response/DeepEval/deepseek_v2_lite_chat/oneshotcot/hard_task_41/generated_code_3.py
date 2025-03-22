import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    block1_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(block1_output)
    block1_output = Flatten()(block1_output)
    block1_output = Dense(units=128, activation='relu')(block1_output)
    block1_output = Dense(units=64, activation='relu')(block1_output)
    block1_output = BatchNormalization()(block1_output)
    block1_output = keras.layers.Dropout(rate=0.5)(block1_output)  # Dropout after batch normalization

    # Concatenate block 1 output with itself after reshaping
    reshaped_block1 = keras.layers.Reshape((64,))(block1_output)
    concatenated_block1 = Concatenate()([block1_output, reshaped_block1])

    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch7 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenate branch outputs
    branch_outputs = [branch1, branch2, branch3, branch4, branch5, branch6, branch7]
    pooled_outputs = [branch4, branch5, branch6]
    concatenated_outputs = Concatenate()(branch_outputs + pooled_outputs)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_outputs)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model