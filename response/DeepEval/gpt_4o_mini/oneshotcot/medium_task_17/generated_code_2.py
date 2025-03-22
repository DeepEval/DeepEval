import keras
from keras.layers import Input, Reshape, Permute, Dense
from keras.models import Model

def dl_model():
    # Step 1: Define input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels

    # Step 2: Reshape the input tensor into (height, width, groups, channels_per_group)
    groups = 3
    channels = 3  # CIFAR-10 has 3 color channels (RGB)
    channels_per_group = channels // groups  # Since groups=3, channels_per_group will be 1
    reshaped = Reshape((32, 32, groups, channels_per_group))(input_layer)  # (height, width, groups, channels_per_group)

    # Step 3: Permute the tensor to shuffle the channels
    permuted = Permute((0, 1, 3, 2))(reshaped)  # Swapping the last two dimensions

    # Step 4: Reshape back to original input shape
    reshaped_back = Reshape((32, 32, 3))(permuted)  # Reshape back to (height, width, channels)

    # Step 5: Fully connected layer for classification
    flatten_layer = keras.layers.Flatten()(reshaped_back)  # Flatten the tensor
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Step 6: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model