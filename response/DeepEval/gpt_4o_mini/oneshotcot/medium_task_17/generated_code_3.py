import keras
from keras.layers import Input, Dense, Permute, Reshape
from keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB)
    input_layer = Input(shape=(32, 32, 3))

    # Reshape input into (height, width, groups, channels_per_group)
    groups = 3
    channels_per_group = 1  # 3 channels total, each group will have 1 channel
    reshaped_tensor = Reshape((32, 32, groups, channels_per_group))(input_layer)

    # Permute to shuffle channels (height, width, channels_per_group, groups)
    permuted_tensor = Permute((1, 2, 4, 3))(reshaped_tensor)

    # Reshape back to original input shape
    reshaped_back_tensor = Reshape((32, 32, 3))(permuted_tensor)

    # Fully connected layer for classification with softmax activation
    # We need to flatten the tensor before passing to Dense layer
    flatten_layer = keras.layers.Flatten()(reshaped_back_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model