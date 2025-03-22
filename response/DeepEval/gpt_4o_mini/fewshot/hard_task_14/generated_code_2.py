import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB

    # Main path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3*3*3, activation='sigmoid')(dense1)  # Output size matches the input layer's channels
    reshaped_weights = Reshape((1, 1, 3))(dense2)  # Reshape to match the number of channels
    weighted_input = Multiply()([input_layer, reshaped_weights])  # Element-wise multiplication

    # Branch path
    branch_conv = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)  # Adjust channels

    # Combining both paths
    combined_output = Add()([weighted_input, branch_conv])

    # Final classification layers
    flatten_layer = Flatten()(combined_output)
    dense3 = Dense(units=256, activation='relu')(flatten_layer)
    dense4 = Dense(units=128, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model