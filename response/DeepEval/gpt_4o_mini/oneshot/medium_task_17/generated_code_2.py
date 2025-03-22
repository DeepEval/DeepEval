import keras
from keras.layers import Input, Dense, Reshape, Permute, Flatten
from keras.models import Model

def dl_model():
    # Define the input layer with shape corresponding to CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Reshape the input tensor into (height, width, groups, channels_per_group)
    # where groups = 3 and channels_per_group = channels / groups = 3 / 3 = 1
    reshaped = Reshape((32, 32, 3, 1))(input_layer)  # Shape becomes (32, 32, 3, 1)

    # Permute the dimensions to enable channel shuffling
    permuted = Permute((1, 2, 4, 3))(reshaped)  # Shape becomes (32, 32, 1, 3)

    # Reshape back to the original input shape (32, 32, 3)
    reshaped_back = Reshape((32, 32, 3))(permuted)  # Shape back to (32, 32, 3)

    # Flatten the output for the fully connected layer
    flatten_layer = Flatten()(reshaped_back)

    # Fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of creating the model
model = dl_model()
model.summary()  # Optional: print the model summary to verify the architecture