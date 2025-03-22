import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1: 3x3 convolution
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1)

    # Branch 2: Max pooling and 3x3 convolution
    branch2_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv1)
    branch2_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_pool)
    branch2_upsample = UpSampling2D(size=(2, 2))(branch2_conv)

    # Branch 3: Max pooling and 3x3 convolution
    branch3_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv1)
    branch3_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_pool)
    branch3_upsample = UpSampling2D(size=(2, 2))(branch3_conv)

    # Concatenate the outputs from the three branches
    main_output = Concatenate()([branch1, branch2_upsample, branch3_upsample])

    # Final convolution for the main path
    main_final_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_output)

    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the outputs from both paths
    combined_output = Add()([main_final_conv, branch_conv])

    # Flatten and fully connected layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model