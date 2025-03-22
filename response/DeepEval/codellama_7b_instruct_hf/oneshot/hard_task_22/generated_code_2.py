import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(units=10, activation='softmax')(main_path)

    # Branch path
    branch_path = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(units=10, activation='softmax')(branch_path)

    # Fuse outputs
    fused_output = Concatenate()([main_path, branch_path])
    fused_output = Flatten()(fused_output)
    fused_output = Dense(units=10, activation='softmax')(fused_output)

    model = keras.Model(inputs=input_layer, outputs=fused_output)
    return model