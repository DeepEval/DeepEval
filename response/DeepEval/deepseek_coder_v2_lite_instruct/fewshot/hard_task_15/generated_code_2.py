import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Reshape, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = GlobalAveragePooling2D()(main_path)
    main_weights = Dense(units=32, activation='relu')(main_path)
    main_weights = Dense(units=32 * 32 * 3, activation='sigmoid')(main_weights)
    main_weights = Reshape(target_shape=(32, 32, 3))(main_weights)
    main_path = Multiply()([main_weights, input_layer])

    # Branch path
    branch_path = input_layer

    # Add outputs from both paths
    combined = Add()([main_path, branch_path])

    # Fully connected layers
    fc1 = Dense(units=128, activation='relu')(combined)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model