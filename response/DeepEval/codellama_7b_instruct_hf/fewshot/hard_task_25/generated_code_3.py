import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = conv1
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    branch3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv1)
    branch1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch1 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(branch1)
    branch2 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(branch2)
    branch3 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(branch3)
    main_path = Add()([branch1, branch2, branch3])

    # Branch path
    conv2 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch4 = conv2
    branch5 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    branch6 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv2)
    branch4 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)
    branch5 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch6 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch6)
    branch4 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(branch4)
    branch5 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(branch5)
    branch6 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(branch6)
    branch_path = Add()([branch4, branch5, branch6])

    # Fuse main and branch path
    fused_path = Add()([main_path, branch_path])

    # Flatten and fully connected layers
    flatten = Flatten()(fused_path)
    output_layer = Dense(10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model