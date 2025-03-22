import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, SeparableConv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Dropout(rate=0.2)(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    branch_path = input_layer
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Dropout(rate=0.2)(branch_path)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)

    # Second block
    groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    group1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(groups[0])
    group1 = Dropout(rate=0.2)(group1)
    group2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
    group2 = Dropout(rate=0.2)(group2)
    group3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(groups[2])
    group3 = Dropout(rate=0.2)(group3)
    output = Concatenate()([group1, group2, group3])

    # Third block
    output = BatchNormalization()(output)
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dropout(rate=0.2)(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dropout(rate=0.2)(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model