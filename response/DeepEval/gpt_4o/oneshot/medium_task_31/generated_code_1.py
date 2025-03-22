import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Concatenate, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Split the input along the channel dimension into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply convolutional layers with different kernel sizes to each split
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the results from different paths
    concatenated = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Flatten the fused features
    flatten_layer = Flatten()(concatenated)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# To create the model, just call the function
model = dl_model()