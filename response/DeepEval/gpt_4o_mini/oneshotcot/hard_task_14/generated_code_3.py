import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Conv2D, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_path = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=512, activation='relu')(main_path)
    dense2_main = Dense(units=3072, activation='sigmoid')(dense1_main)  # 32 * 32 * 3 = 3072
    reshaped_weights = keras.layers.Reshape((1, 1, 3072))(dense2_main)  # Reshape to match input for multiplication
    scaled_main = keras.layers.Multiply()([input_layer, reshaped_weights])

    # Branch Path
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine both paths
    combined_output = Add()([scaled_main, branch_path])

    # Fully connected layers after combining
    flatten_layer = Flatten()(combined_output)
    dense1_combined = Dense(units=256, activation='relu')(flatten_layer)
    dense2_combined = Dense(units=128, activation='relu')(dense1_combined)
    output_layer = Dense(units=10, activation='softmax')(dense2_combined)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model