import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, Resizing
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch 2 (downsample and then upsample)
    branch2_downsample = MaxPooling2D(pool_size=(2, 2), strides=2)(main_path)
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_downsample)
    branch2_upsample = Resizing(height=32, width=32, interpolation="bilinear")(branch2_conv)

    # Branch 3 (downsample and then upsample)
    branch3_downsample = MaxPooling2D(pool_size=(2, 2), strides=2)(main_path)
    branch3_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3_downsample)
    branch3_upsample = Resizing(height=32, width=32, interpolation="bilinear")(branch3_conv)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([main_path, branch1, branch2_upsample, branch3_upsample])

    # 1x1 convolutional layer for final output
    final_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Addition of main path and branch path
    added = Add()([final_output, branch_path])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(added)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()