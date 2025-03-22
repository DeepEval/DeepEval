import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, DepthwiseConv2D, LayerNormalization

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Depthwise separable convolution layer
    dw_conv = DepthwiseConv2D(kernel_size=(7, 7), depth_multiplier=1, strides=(2, 2),
                              depthwise_constraint=keras.constraints.max_norm(1.))(input_layer)
    dw_conv = BatchNormalization(axis=3, momentum=0.99, epsilon=1e-3)(dw_conv)
    dw_conv = LayerNormalization(axis=3)(dw_conv)
    dw_conv = Activation('relu')(dw_conv)

    # Fully connected layer for channel-wise feature transformation
    fc1 = Dense(units=128, activation='relu')(dw_conv)
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc2)

    # Combine original input and processed features
    combined_output = Concatenate()([input_layer, dw_conv])

    # Return the constructed model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Print the model summary
model = dl_model()
model.summary()