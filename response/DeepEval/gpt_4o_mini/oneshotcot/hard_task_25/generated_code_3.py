import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Step 2: Main path 1x1 convolution
    main_path_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Split into three branches
    # First branch - 3x3 convolution
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_conv1)

    # Second branch - average pooling + 3x3 convolution + transpose convolution
    branch2_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path_conv1)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2_conv)

    # Third branch - average pooling + 3x3 convolution + transpose convolution
    branch3_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path_conv1)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3_conv)

    # Step 4: Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # Step 5: Apply 1x1 convolution to form the main path output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Step 6: Branch path with 1x1 convolution to match channels
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 7: Fuse main and branch path outputs through addition
    added_output = Add()([main_path_output, branch_path])

    # Step 8: Batch normalization
    batch_norm = BatchNormalization()(added_output)

    # Step 9: Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Step 10: Fully connected layer for 10-class classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model