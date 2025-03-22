import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, concatenate


def dl_model():
    # Input layer for the image data
    input_img = Input(shape=(28, 28, 1))

    # Block 1: Parallel Max Pooling Paths
    path_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_img)
    path_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_img)
    path_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_img)

    # Flatten the outputs of the parallel paths
    path_1 = Flatten()(path_1)
    path_2 = Flatten()(path_2)
    path_3 = Flatten()(path_3)

    # Regularize the flattened vectors
    path_1 = Dropout(0.25)(path_1)
    path_2 = Dropout(0.25)(path_2)
    path_3 = Dropout(0.25)(path_3)

    # Concatenate the regularized vectors
    concat_layer = concatenate([path_1, path_2, path_3])

    # Reshape the concatenated vector for Block 2
    reshaped_vector = Reshape((1, 1, -1))(concat_layer)

    # Block 2: Parallel Convolution and Pooling Paths
    path_1 = Conv2D(filters=8, kernel_size=(1, 1), activation='relu')(reshaped_vector)
    path_2 = Conv2D(filters=8, kernel_size=(1, 1), activation='relu')(reshaped_vector)
    path_2 = Conv2D(filters=16, kernel_size=(1, 7), activation='relu')(path_2)
    path_2 = Conv2D(filters=16, kernel_size=(7, 1), activation='relu')(path_2)
    path_3 = Conv2D(filters=8, kernel_size=(1, 1), activation='relu')(reshaped_vector)
    path_3 = Conv2D(filters=16, kernel_size=(7, 1), activation='relu')(path_3)
    path_3 = Conv2D(filters=16, kernel_size=(1, 7), activation='relu')(path_3)
    path_4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(reshaped_vector)
    path_4 = Conv2D(filters=8, kernel_size=(1, 1), activation='relu')(path_4)

    # Concatenate the outputs of Block 2
    concat_layer_2 = concatenate([path_1, path_2, path_3, path_4])

    # Fully connected layers for classification
    dense_layer_1 = Dense(units=128, activation='relu')(concat_layer_2)
    output_layer = Dense(units=10, activation='softmax')(dense_layer_1)

    # Create the model
    model = Model(inputs=input_img, outputs=output_layer)

    return model