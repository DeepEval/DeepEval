import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense


def dl_model():
    # Define the input layer with shape (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Define the first residual block
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    branch = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Add()([x, branch])

    # Define the second residual block
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    branch = Conv2D(64, (3, 3), activation='relu')(x)
    x = Add()([x, branch])

    # Define the third residual block
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    branch = Conv2D(128, (3, 3), activation='relu')(x)
    x = Add()([x, branch])

    # Define the output layer with shape (10,)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model