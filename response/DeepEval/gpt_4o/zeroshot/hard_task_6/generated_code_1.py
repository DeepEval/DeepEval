import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Concatenate, GlobalAveragePooling2D, Dense, DepthwiseConv2D, Reshape, Permute
from tensorflow.keras.models import Model

def block1(x):
    # Split the input into three groups
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x)
    
    # Apply a 1x1 Conv layer to each group
    convs = [Conv2D(x.shape[-1] // 3, (1, 1), activation='relu')(s) for s in split]
    
    # Concatenate the outputs
    out = Concatenate()(convs)
    return out

def block2(x):
    # Obtain input shape
    input_shape = tf.shape(x)
    height, width, channels = input_shape[1], input_shape[2], input_shape[3]
    channels_per_group = channels // 3
    
    # Reshape
    reshaped = Reshape((height, width, 3, channels_per_group))(x)
    
    # Permute third and fourth dimensions for channel shuffling
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to original shape
    out = Reshape((height, width, channels))(permuted)
    return out

def block3(x):
    # Apply a 3x3 Depthwise Separable Convolution
    out = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, activation='relu')(x)
    return out

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3
    
    # Main Path
    x = block1(input_layer)
    x = block2(x)
    x = block3(x)
    x = block1(x)  # Repeating Block 1
    
    # Branch Path
    branch = GlobalAveragePooling2D()(input_layer)
    
    # Concatenate Main Path and Branch Path
    concatenated = Concatenate()([x, branch])
    
    # Fully connected layer for classification
    output = Dense(10, activation='softmax')(concatenated)  # CIFAR-10 has 10 classes
    
    model = Model(inputs=input_layer, outputs=output)
    return model

# Example of creating and summarizing the model
model = dl_model()
model.summary()