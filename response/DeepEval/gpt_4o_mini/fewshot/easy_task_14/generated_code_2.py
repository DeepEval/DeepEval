import keras
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 1: Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Step 2: Fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Generate weights for the 3 channels

    # Step 3: Reshape weights to match the input feature map's channels
    reshaped_weights = Reshape((1, 1, 3))(dense2)  # Reshape to (1, 1, 3) to apply on 32x32 feature map

    # Step 4: Multiply element-wise with the input feature map
    weighted_input = Multiply()([input_layer, reshaped_weights])

    # Step 5: Flatten the result
    flatten_layer = Flatten()(weighted_input)

    # Step 6: Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model