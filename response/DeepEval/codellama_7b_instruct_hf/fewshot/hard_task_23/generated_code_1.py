import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Local feature extraction branch
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Downsampling branch 1
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Downsampling branch 2
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch2)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Upsampling branch
    branch4 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(branch3)

    # Concatenate branches
    merged_layer = Concatenate()([branch1, branch2, branch3, branch4])

    # Refine features using 1x1 convolutional layer
    refined_layer = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(merged_layer)

    # Flatten layer
    flatten_layer = Flatten()(refined_layer)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model