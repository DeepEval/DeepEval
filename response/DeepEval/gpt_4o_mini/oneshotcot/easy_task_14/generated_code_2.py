import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    # Step 1: Add input layer for CIFAR-10 images (32x32 pixels, 3 color channels)
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Apply global average pooling to compress the input features
    gap = GlobalAveragePooling2D()(input_layer)

    # Step 3: First fully connected layer
    dense1 = Dense(units=128, activation='relu')(gap)
    
    # Step 4: Second fully connected layer
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Step 5: Generate weights that match the number of channels in the input (3 channels)
    weights = Dense(units=3, activation='sigmoid')(dense2)  # Sigmoid to ensure weights are in [0, 1]

    # Step 6: Reshape the weights to match the input shape
    reshaped_weights = Reshape((1, 1, 3))(weights)  # Reshape to (1, 1, 3) for broadcasting

    # Step 7: Multiply element-wise with the input feature map
    scaled_input = Multiply()([input_layer, reshaped_weights])

    # Step 8: Flatten the result
    flatten_layer = Flatten()(scaled_input)

    # Step 9: Final fully connected layer to output the probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model