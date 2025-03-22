import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense, BatchNormalization

def dl_model():     
    # Step 1: Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels and have 3 color channels

    # Step 2: Initial 1x1 Convolutional Layer
    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(initial_conv)

    # Step 4: Branch 2
    branch2_pool = MaxPooling2D(pool_size=(2, 2), padding='same')(initial_conv)
    branch2_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2_pool)
    branch2_up = UpSampling2D(size=(2, 2))(branch2_conv)

    # Step 5: Branch 3
    branch3_pool = MaxPooling2D(pool_size=(2, 2), padding='same')(initial_conv)
    branch3_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch3_pool)
    branch3_up = UpSampling2D(size=(2, 2))(branch3_conv)

    # Step 6: Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2_up, branch3_up])

    # Step 7: Another 1x1 Convolutional Layer
    conv_final = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(concatenated)

    # Step 8: Flatten layer
    flatten_layer = Flatten()(conv_final)

    # Step 9: Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)  # 10 classes for CIFAR-10

    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model