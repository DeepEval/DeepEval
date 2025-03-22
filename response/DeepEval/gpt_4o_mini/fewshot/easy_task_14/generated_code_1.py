import keras
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 1: Global Average Pooling
    gap_output = GlobalAveragePooling2D()(input_layer)

    # Step 2: Fully Connected Layers to Learn Channel Weights
    dense1 = Dense(units=128, activation='relu')(gap_output)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Output shape matching the number of channels (3 for RGB)

    # Step 3: Reshape Weights to Match Input Shape
    reshaped_weights = Reshape((1, 1, 3))(dense2)  # Reshape to (1, 1, 3) for broadcasting

    # Step 4: Element-wise Multiplication with Input Feature Map
    weighted_output = Multiply()([input_layer, reshaped_weights])

    # Step 5: Flatten and Fully Connected Layer for Final Output
    flatten_output = Flatten()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # 10 classes for CIFAR-10

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model