import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path with global average pooling and two dense layers
    gap = GlobalAveragePooling2D()(input_layer)
    dense1_main = Dense(units=32, activation='relu')(gap)
    dense2_main = Dense(units=3, activation='sigmoid')(dense1_main)  # Assuming output channels are 3
    weights = Multiply()([input_layer, dense2_main])

    # Branch path with a 3x3 convolution
    conv_branch = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Combine the main path and branch path
    combined = Add()([weights, conv_branch])

    # Pass through three fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model