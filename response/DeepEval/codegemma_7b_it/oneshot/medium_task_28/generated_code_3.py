import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Reshape, GlobalAveragePooling2D, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Generate attention weights
    attention = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(input_layer)
    attention = Reshape((32*32,))(attention)
    attention = Flatten()(attention)
    attention = Dense(units=128, activation='relu')(attention)
    attention = Dense(units=32*32, activation='softmax')(attention)

    # Multiply attention weights with input features
    weighted_input = Multiply()([attention, Reshape((32, 32, 1))(input_layer)])
    weighted_input = Reshape((32, 32, 3))(weighted_input)

    # Reduce dimensionality and restore it
    contextual_info = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_input)
    contextual_info = BatchNormalization()(contextual_info)
    contextual_info = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(contextual_info)

    # Add contextual information to input image
    processed_output = Add()([input_layer, contextual_info])

    # Flattened and fully connected layers
    flatten_layer = Flatten()(processed_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model