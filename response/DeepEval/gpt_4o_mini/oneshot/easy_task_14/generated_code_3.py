import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Global average pooling to capture global information
    global_avg_pool = GlobalAveragePooling2D()(input_layer)

    # Two fully connected layers to generate weights
    dense1 = Dense(units=256, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3 * 32 * 32, activation='sigmoid')(dense1)  # 3 channels for CIFAR-10

    # Reshape the output to match the input feature map shape
    reshaped_weights = Reshape((1, 1, 3 * 32 * 32))(dense2)  # Reshape for element-wise multiplication

    # Multiply the reshaped weights with the input feature map
    multiplied_output = Multiply()([input_layer, reshaped_weights])

    # Flatten the result
    flatten_layer = Flatten()(multiplied_output)

    # Fully connected layer for final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
model.summary()  # This will print a summary of the model architecture