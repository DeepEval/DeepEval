from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, 
                         concatenate, Activation, Flatten, Dense)
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

def dl_model():
    # Input image
    input_img = Input(shape=(32, 32, 3))

    # Feature extraction path 1: 1x1 convolution
    path_1 = Conv2D(filters=16, kernel_size=1, strides=(1, 1), padding='same')(input_img)
    path_1 = Activation('relu')(path_1)
    path_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path_1)
    path_1 = Dropout(0.25)(path_1)

    # Feature extraction path 2: 1x1, 1x7, 7x1 convolutions
    path_2 = Conv2D(filters=16, kernel_size=1, strides=(1, 1), padding='same')(input_img)
    path_2 = Activation('relu')(path_2)
    path_2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same')(path_2)
    path_2 = Activation('relu')(path_2)
    path_2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same')(path_2)
    path_2 = Activation('relu')(path_2)
    path_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path_2)
    path_2 = Dropout(0.25)(path_2)

    # Concatenate outputs from both paths and apply 1x1 convolution
    concat_layer = concatenate([path_1, path_2])
    concat_layer = Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='same')(concat_layer)
    concat_layer = Activation('relu')(concat_layer)

    # Branch directly connected to input
    branch_layer = Conv2D(filters=64, kernel_size=1, strides=(1, 1), padding='same')(input_img)
    branch_layer = Activation('relu')(branch_layer)

    # Merge outputs of main path and branch through addition
    merged_layer = add([concat_layer, branch_layer])
    merged_layer = Activation('relu')(merged_layer)
    merged_layer = Dropout(0.25)(merged_layer)

    # Classification layers
    x = Flatten()(merged_layer)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Model definition
    model = Model(inputs=input_img, outputs=predictions)
    model.compile(optimizer=SGD(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    return model