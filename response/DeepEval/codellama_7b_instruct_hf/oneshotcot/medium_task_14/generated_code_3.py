import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Sequential block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm1)

    # Sequential block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    batch_norm2 = BatchNormalization()(conv2)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm2)

    # Sequential block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    batch_norm3 = BatchNormalization()(conv3)
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm3)

    # Parallel branch
    parallel_branch = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    parallel_branch = BatchNormalization()(parallel_branch)

    # Add outputs from sequential blocks and parallel branch
    output1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm1)
    output2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm2)
    output3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm3)
    parallel_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(parallel_branch)
    output = Concatenate()([output1, output2, output3, parallel_output])

    # Flatten output
    flattened = Flatten()(output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model