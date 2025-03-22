import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_output = input_layer
    for kernel_size in [1, 3, 5]:
        main_output = Lambda(lambda x: tf.split(x, 3, axis=-1))(main_output)
        main_output = Concatenate(axis=-1)([
            Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', activation='relu')(x)
            for x in main_output
        ])
        main_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_output)
    main_output = BatchNormalization()(main_output)
    main_output = Flatten()(main_output)

    # Branch Path
    branch_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_output)
    branch_output = Flatten()(branch_output)

    # Fusion
    combined_output = Concatenate()([main_output, branch_output])

    # Classification Layers
    dense1 = Dense(units=128, activation='relu')(combined_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model