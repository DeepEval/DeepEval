import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=3 * 32 * 32, activation='relu')(main_path)
    main_path = Dense(units=3 * 32 * 32, activation='relu')(main_path)
    main_path = Reshape(target_shape=(32, 32, 3))(main_path)

    # Branch path
    branch_path = input_layer

    # Combine paths
    combined_path = Add()([main_path, branch_path])

    # Final fully connected layers
    output_layer = Dense(units=10, activation='softmax')(combined_path)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model