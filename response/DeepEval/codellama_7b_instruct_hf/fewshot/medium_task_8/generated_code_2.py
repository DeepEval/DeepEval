import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = MaxPooling2D(pool_size=(2, 2))(main_path)
    main_path = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(main_path)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = Add()([main_path, main_path[1]])
    main_path = Flatten()(main_path)

    # Branch path
    branch_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2))(branch_path)
    branch_path = Add()([branch_path, branch_path])
    branch_path = Flatten()(branch_path)

    # Combine outputs
    output = Add()([main_path, branch_path])
    output = Flatten()(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model