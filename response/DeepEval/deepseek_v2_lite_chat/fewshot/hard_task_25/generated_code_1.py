import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Conv2DTranspose, AveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    main_pool = AveragePooling2D(pool_size=(2, 2), padding='same')(main_conv)

    # Branch path
    branch_conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_pool1 = AveragePooling2D(pool_size=(2, 2), padding='same')(branch_conv1)
    branch_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch_pool1)
    transpose_conv1 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(branch_conv2)
    transpose_conv2 = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding='same')(transpose_conv1)

    # Concatenate outputs of main path and branch path
    concat_layer = Concatenate()([main_pool, transpose_conv2])

    # Additional convolutional layers
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(concat_layer)
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv3)

    # Fully connected layer
    flatten = Flatten()(conv4)
    dense = Dense(units=512, activation='relu')(flatten)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()