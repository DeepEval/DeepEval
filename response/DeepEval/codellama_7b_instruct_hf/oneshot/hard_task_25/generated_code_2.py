from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Main path
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    branch3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    main_path = Concatenate()([branch1, branch2, branch3, branch4])

    # Branch path
    branch_layer = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fusion
    fusion = main_path + branch_layer

    # Output
    output = Flatten()(fusion)
    output = Dense(units=10, activation='softmax')(output)

    # Model
    model = Model(inputs=input_layer, outputs=output)

    return model