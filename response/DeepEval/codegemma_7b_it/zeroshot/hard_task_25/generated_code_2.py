from keras.layers import Input, Conv2D, BatchNormalization, Activation, concatenate, add, AveragePooling2D, UpSampling2D, Flatten, Dense
from keras.models import Model

def dl_model():

    # Input layer
    img = Input(shape=(32, 32, 3))

    # Main path
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(img)
    conv_1x1 = BatchNormalization()(conv_1x1)
    conv_1x1 = Activation('relu')(conv_1x1)

    # Branch path
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv_1x1)
    branch_1 = BatchNormalization()(branch_1)
    branch_1 = Activation('relu')(branch_1)

    # Downsampling for branch path
    branch_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1x1)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch_2)
    branch_2 = BatchNormalization()(branch_2)
    branch_2 = Activation('relu')(branch_2)

    # Downsampling for branch path
    branch_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1x1)
    branch_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch_3)
    branch_3 = BatchNormalization()(branch_3)
    branch_3 = Activation('relu')(branch_3)
    branch_3 = UpSampling2D(size=(2, 2))(branch_3)

    # Upsampling for branch path
    branch_2 = UpSampling2D(size=(2, 2))(branch_2)

    # Concatenating outputs of branches
    concat = concatenate([branch_1, branch_2, branch_3])

    # Main path output
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)
    main_path_output = BatchNormalization()(main_path_output)
    main_path_output = Activation('relu')(main_path_output)

    # Branch path output
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)

    # Fusing outputs
    output = add([main_path_output, branch_path_output])

    # Fully connected layer for classification
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    # Model definition
    model = Model(inputs=img, outputs=output)

    return model