import keras
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 1: Global Average Pooling to capture global information
    pooled_output = GlobalAveragePooling2D()(input_layer)

    # Step 2: Two fully connected layers to generate weights for the channels
    dense1 = Dense(units=128, activation='relu')(pooled_output)
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Output size equals the number of channels

    # Step 3: Reshape the weights to match the input shape (32, 32, 3)
    reshaped_weights = Dense(units=32 * 32 * 3, activation='sigmoid')(dense2)
    reshaped_weights = tf.reshape(reshaped_weights, shape=(-1, 32, 32, 3))

    # Step 4: Element-wise multiplication with the original input feature map
    weighted_output = Multiply()([input_layer, reshaped_weights])

    # Step 5: Flatten the result and pass through a final fully connected layer for classification
    flatten_output = Flatten()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)  # CIFAR-10 has 10 classes

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model