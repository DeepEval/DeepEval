import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # Step 1: Global Average Pooling to capture global information
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Step 2: Two fully connected layers to generate weights
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)  # Adjust units as needed
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Output units should match the input channels

    # Step 3: Reshape weights to align with input shape
    channel_weights = Reshape((1, 1, 3))(dense2)

    # Step 4: Element-wise multiplication with input feature map
    scaled_features = Multiply()([input_layer, channel_weights])

    # Step 5: Flatten and pass through a fully connected layer for final classification
    flatten_layer = Flatten()(scaled_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model