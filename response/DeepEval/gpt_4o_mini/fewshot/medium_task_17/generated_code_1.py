import keras
import tensorflow as tf
from keras.layers import Input, Reshape, Permute, Dense, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels (RGB)

    # Reshape input tensor to (height, width, groups, channels_per_group)
    # Here, groups=3 and channels_per_group=1 (since each RGB channel will be treated as a separate group)
    reshaped = Reshape(target_shape=(32, 32, 3, 1))(input_layer)

    # Permute the dimensions to swap the channels and groups
    # From (height, width, groups, channels_per_group) to (height, width, channels_per_group, groups)
    permuted = Permute((0, 1, 3, 2))(reshaped)

    # Reshape back to original input shape
    reshaped_back = Reshape(target_shape=(32, 32, 3))(permuted)

    # Flatten the tensor before passing it to the fully connected layer
    flatten = Flatten()(reshaped_back)

    # Fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model