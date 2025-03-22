import keras
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Permute, Lambda
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels (RGB)
    
    # Reshape input tensor to (height, width, groups, channels_per_group)
    # Here, we have groups=3 and channels_per_group=1 (since 3 channels / 3 groups = 1)
    reshaped = Reshape((32, 32, 3, 1))(input_layer)
    
    # Permute the dimensions to shuffle the channels
    permuted = Permute((0, 1, 3, 2))(reshaped)  # (height, width, channels_per_group, groups)

    # Reshape back to the original input shape
    reshaped_back = Reshape((32, 32, 3))(permuted)
    
    # Flatten the reshaped tensor for the fully connected layer
    flatten = tf.keras.layers.Flatten()(reshaped_back)
    
    # Fully connected layer with softmax activation for classification into 10 classes
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model