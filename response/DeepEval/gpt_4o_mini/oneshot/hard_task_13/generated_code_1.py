import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Multiply, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 channels (RGB)

    # First block with parallel branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate outputs from all paths
    block1_output = Concatenate()([path1, path2, path3, path4])

    # Second block with Global Average Pooling
    global_avg_pooling = GlobalAveragePooling2D()(block1_output)

    # Fully connected layers to generate weights
    dense1 = Dense(units=block1_output.shape[-1], activation='relu')(global_avg_pooling)  # Same size as channels of block 2 output
    dense2 = Dense(units=block1_output.shape[-1], activation='sigmoid')(dense1)  # Producing weights

    # Reshape the weights to match the shape of block1_output's channels
    reshaped_weights = Reshape((1, 1, block1_output.shape[-1]))(dense2)

    # Element-wise multiplication with the block1_output feature maps
    multiplied_output = Multiply()([block1_output, reshaped_weights])

    # Final fully connected layer for output
    final_output = GlobalAveragePooling2D()(multiplied_output)  # Pooling to prepare for final dense layer
    output_layer = Dense(units=10, activation='softmax')(final_output)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model