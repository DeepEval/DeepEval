import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Reshape, Concatenate, Add, BatchNormalization
from keras.models import Model

def dl_model():

    # Block 1
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    main_path = x

    # Transition Convolution
    x = Conv2D(128, (1, 1), activation='relu')(main_path)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    branch_path = x

    # Block 2
    x = GlobalAveragePooling2D()(branch_path)
    x = Reshape((1, 1, 128))(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Reshape((1, 1, 128))(x)
    x = Multiply()([main_path, x])
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model