import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply
from keras.models import Model

def dl_model():

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1: Parallel branches
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate the outputs of the parallel branches
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Block 2: Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(concatenated)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=concatenated.shape[-1], activation='sigmoid')(dense1)  # Generate weights for reshaping

    # Reshape the weights to match the input shape of Block 2
    reshaped_weights = Reshape(target_shape=(1, 1, concatenated.shape[-1]))(dense2)

    # Multiply the reshaped weights with the concatenated feature maps (element-wise)
    weighted_output = Multiply()([concatenated, reshaped_weights])

    # Final flattening and output layer
    flatten_output = GlobalAveragePooling2D()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model