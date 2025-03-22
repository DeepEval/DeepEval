import keras
from keras.layers import Input, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Concatenate, SeparableConv2D, ReLU
from keras.models import Model

def dl_model():
    # Main path
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    main_path = Model(inputs=input_layer, outputs=x)

    # Branch path
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    branch_path = Model(inputs=input_layer, outputs=x)

    # Combine main and branch paths
    x = main_path(input_layer)
    x = Concatenate()([x, branch_path(input_layer)])
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x)

    return model