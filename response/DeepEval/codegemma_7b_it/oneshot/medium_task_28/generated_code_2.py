import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Conv2DTranspose, Lambda, Multiply

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Generate attention weights
    attention_conv = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    attention_weights = Flatten()(attention_conv)
    attention_weights = Dense(units=32 * 32, activation='softmax')(attention_weights)
    attention_weights = Reshape((32, 32, 1))(attention_weights)

    # Multiply attention weights with input features
    weighted_features = Multiply()([input_layer, attention_weights])

    # Reduce dimensionality
    reduction_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_features)
    reduction_conv = BatchNormalization()(reduction_conv)
    reduction_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reduction_conv)

    # Add to original input
    output = Add()([input_layer, reduction_conv])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model