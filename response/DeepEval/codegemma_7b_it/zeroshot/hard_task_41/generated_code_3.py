import keras
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Dense, Reshape, concatenate, Lambda, GlobalAveragePooling2D
from keras.regularizers import l2

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path3 = Flatten()(path3)

    path1 = Dropout(0.2)(path1)
    path2 = Dropout(0.2)(path2)
    path3 = Dropout(0.2)(path3)

    concat_layer = concatenate([path1, path2, path3])

    # Block 2
    reshape_layer = Reshape((1, 1, -1))(concat_layer)
    fc_layer = Dense(256, activation='relu')(reshape_layer)

    # Branch connections
    branch1 = Conv2D(64, (1, 1), padding='same')(fc_layer)
    branch2 = Conv2D(64, (1, 1), padding='same')(fc_layer)
    branch2 = Conv2D(128, (3, 3), padding='same')(branch2)
    branch3 = Conv2D(64, (1, 1), padding='same')(fc_layer)
    branch3 = Conv2D(128, (3, 3), padding='same')(branch3)
    branch3 = Conv2D(128, (3, 3), padding='same')(branch3)
    branch4 = GlobalAveragePooling2D()(fc_layer)
    branch4 = Conv2D(64, (1, 1), padding='same')(branch4)

    # Concatenate and fuse
    concat_branch = concatenate([branch1, branch2, branch3, branch4])
    fusion_layer = Conv2D(128, (1, 1), padding='same')(concat_branch)

    # Output layers
    flatten_layer = Flatten()(fusion_layer)
    output_layer = Dense(10, activation='softmax')(flatten_layer)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model