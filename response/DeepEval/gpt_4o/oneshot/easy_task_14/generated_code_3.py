import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten
from keras.models import Model
import keras.backend as K

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3

    # Global Average Pooling to capture global information
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # First fully connected layer to learn channel-wise weights
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)

    # Second fully connected layer to generate weights same size as the channels
    dense2 = Dense(units=3, activation='sigmoid')(dense1)  # Assuming the number of input channels is 3

    # Reshape weights to match the input shape (1, 1, 3)
    weights = Reshape((1, 1, 3))(dense2)

    # Multiply input feature map with the learned weights
    weighted_input = Multiply()([input_layer, weights])

    # Flatten the result
    flatten_layer = Flatten()(weighted_input)

    # Final fully connected layer to produce the output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model