import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ZeroPadding2D, Conv2DTranspose, Flatten, Dense

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32
    input_layer = Input(shape=input_shape)

    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)

    # Three branches
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch1)
    branch1 = UpSampling2D(size=2)(branch1)

    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D(size=2)(branch2)

    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D(size=2)(branch3)

    # Concatenate all branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Final 1x1 convolutional layer
    conv_out = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)

    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(conv_out)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()