import keras
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(64, (7, 7), padding='same', activation='relu')(input_layer)
    main_path = BatchNormalization()(main_path)
    main_path = MaxPooling2D((2, 2), padding='same')(main_path)
    main_path = Conv2D(64, (1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(64, (1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Combine the two paths
    output = Add()([main_path, branch_path])

    # Flatten and classify
    output = Flatten()(output)
    output = Dense(10, activation='softmax')(output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model