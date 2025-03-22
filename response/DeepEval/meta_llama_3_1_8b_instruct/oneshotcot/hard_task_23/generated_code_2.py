import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense, Transpose

def dl_model():
    # Define the input layer with shape (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Add a 1x1 initial convolutional layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Define the local feature extraction branch
    local_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)
    local_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(local_branch)

    # Step 3: Define the downsampling branch 1
    downsampling_branch1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    downsampling_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(downsampling_branch1)

    # Step 4: Define the upsampling branch 1
    upsampling_branch1 = UpSampling2D(size=(2, 2))(downsampling_branch1)
    upsampling_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(upsampling_branch1)

    # Step 5: Define the downsampling branch 2
    downsampling_branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(initial_conv)
    downsampling_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(downsampling_branch2)

    # Step 6: Define the upsampling branch 2
    upsampling_branch2 = UpSampling2D(size=(2, 2))(downsampling_branch2)
    upsampling_branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(upsampling_branch2)

    # Step 7: Concatenate the outputs of the branches
    concatenated_output = Concatenate()([local_branch, upsampling_branch1, upsampling_branch2])

    # Step 8: Add a 1x1 convolutional layer for refinement
    refined_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated_output)

    # Step 9: Flatten the output
    flattened_output = Flatten()(refined_output)

    # Step 10: Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model