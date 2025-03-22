import keras
from keras.layers import Input, Conv2D, LayerNormalization, Conv2DTranspose, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Depthwise convolution
        depthwise_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', depth_kw={}, depth_ptr=None)(input_tensor)
        depthwise_norm = LayerNormalization(scale=False)(depthwise_conv)
        
        # First 1x1 pointwise convolution
        pointwise_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_norm)
        
        # Second 1x1 pointwise convolution
        pointwise_conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pointwise_conv1)

        # Branch path
        branch_output = input_tensor

        # Combine outputs
        combined_output = Add()([depthwise_norm, pointwise_conv2, branch_output])

        # Flatten and pass through fully connected layers
        flatten = Flatten()(combined_output)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    # Construct the model
    model = main_path(input_layer)

    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()