import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=64, activation='relu')(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)
    weights = Dense(units=32, activation='relu')(main_path)
    weights = Reshape((1, 1, 32))(weights)
    main_path = Multiply()([input_layer, weights])

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)

    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Flatten the combined result
    flatten_layer = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model