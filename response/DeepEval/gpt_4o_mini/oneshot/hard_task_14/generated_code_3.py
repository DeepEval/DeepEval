import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Conv2D, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    pooled_output = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=512, activation='relu')(pooled_output)
    dense2 = Dense(units=256, activation='relu')(dense1)
    weights = Dense(units=3, activation='sigmoid')(dense2)  # Output size matches the number of channels (3)

    # Reshape weights to match the input layer's shape
    weights_reshaped = Reshape((1, 1, 3))(weights)  # Reshape to (1, 1, 3) for broadcasting
    scaled_input = Multiply()([input_layer, weights_reshaped])  # Element-wise multiplication

    # Branch path
    branch_output = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine paths
    combined_output = Add()([scaled_input, branch_output])

    # Final classification layers
    flatten_layer = Flatten()(combined_output)
    dense3 = Dense(units=256, activation='relu')(flatten_layer)
    dense4 = Dense(units=128, activation='relu')(dense3)
    output_layer = Dense(units=10, activation='softmax')(dense4)  # 10 classes for CIFAR-10

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model