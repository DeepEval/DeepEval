import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Conv2D, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=32, activation='relu')(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)
    main_path = Reshape(target_shape=(32, 32, 3))(main_path) 

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add the outputs
    combined = Add()([main_path, branch_path])

    # Final classification layers
    flatten = Flatten()(combined)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model