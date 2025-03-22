import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block with parallel branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate all paths
    block1_output = Concatenate()([path1, path2, path3, path4])

    # Second block: Global average pooling
    pooled_output = GlobalAveragePooling2D()(block1_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(pooled_output)
    dense2 = Dense(units=block1_output.shape[-1], activation='sigmoid')(dense1)  # Ensure output channels match

    # Reshape to match input feature map shape
    reshaped_weights = Reshape((1, 1, block1_output.shape[-1]))(dense2)

    # Element-wise multiplication with the input feature map
    feature_map = Multiply()([block1_output, reshaped_weights])

    # Final fully connected layer for output
    output_layer = Dense(units=10, activation='softmax')(feature_map)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model