import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Concatenate()([x, x, x])
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Block 2
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Concatenate()([x, x, x])
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Block 3
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Concatenate()([x, x, x])
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(branch_path)
    branch_path = Flatten()(branch_path)

    # Combine the outputs from both the main path and the branch path
    x = Concatenate()([x, branch_path])

    # Fully connected layer
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model