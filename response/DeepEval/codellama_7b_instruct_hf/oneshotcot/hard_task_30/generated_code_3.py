import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Split

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = input_layer

    # Second block
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(branch_path)
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    group3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    concatenated_groups = Concatenate()([group1, group2, group3])

    # Model
    model = keras.Model(inputs=input_layer, outputs=concatenated_groups)
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))

    return model