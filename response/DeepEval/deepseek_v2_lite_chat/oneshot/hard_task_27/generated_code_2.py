import keras
from keras.layers import Input, Conv2D, LayerNormalization, Add, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Depthwise separable convolution
    conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same',
                  kernel_constraint=keras.constraints.MaxNorm(3.))(input_layer)
    conv = LayerNormalization()(conv)  # Layer normalization

    # Two fully connected paths for channel-wise feature transformation
    dense1 = Dense(units=128, activation='relu')(conv)
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Add original input to transformed features
    combined = Add()([input_layer, dense2])

    # Final classification layer
    flatten = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()