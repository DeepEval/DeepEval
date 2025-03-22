import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Softmax, Multiply, LayerNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 1: Generate attention weights with 1x1 convolution
    attention_weights = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    attention_weights = Softmax()(attention_weights)

    # Step 2: Multiply input features with attention weights
    weighted_input = Multiply()([input_layer, attention_weights])

    # Step 3: Reduce input dimensionality using another 1x1 convolution
    reduced_dimensionality = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(weighted_input)

    # Step 4: Apply layer normalization and ReLU activation
    normalized = LayerNormalization()(reduced_dimensionality)
    activated = ReLU()(normalized)

    # Step 5: Restore dimensionality with another 1x1 convolution
    restored_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(activated)

    # Step 6: Add processed output to the original input image
    final_output = Add()([input_layer, restored_output])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model