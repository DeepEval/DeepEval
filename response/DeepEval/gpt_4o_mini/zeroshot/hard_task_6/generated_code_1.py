import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, AveragePooling2D, Concatenate, Dense, Lambda, Reshape, Permute, DepthwiseConv2D
from tensorflow.keras.models import Model

def block1(inputs):
    # Split the input into three groups
    groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    processed_groups = []
    
    for group in groups:
        x = Conv2D(filters=inputs.shape[-1] // 3, kernel_size=(1, 1))(group)
        x = Activation('relu')(x)
        processed_groups.append(x)
    
    # Concatenate the processed groups to produce fused features
    return Concatenate(axis=-1)(processed_groups)

def block2(inputs):
    # Get the shape of the features from Block 1
    shape = tf.shape(inputs)
    height, width = shape[1], shape[2]
    channels = shape[-1]
    groups = 3
    channels_per_group = channels // groups
    
    # Reshape the features into (height, width, groups, channels_per_group)
    x = Reshape((height, width, groups, channels_per_group))(inputs)
    # Permute to swap the last two dimensions
    x = Permute((1, 2, 4, 3))(x)
    # Reshape back to original shape
    return Reshape((height, width, channels))(x)

def block3(inputs):
    # Apply a 3x3 depthwise separable convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(inputs)
    x = Activation('relu')(x)
    return x

def dl_model():
    inputs = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB

    # Main path
    x = block1(inputs)
    x = block2(x)
    x = block3(x)
    x = block1(x)  # Repeating Block 1

    # Branch path
    branch = AveragePooling2D(pool_size=(2, 2))(inputs)
    branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch)

    # Concatenate main path and branch path
    x = Concatenate()([x, branch])

    # Fully connected layer for classification
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(units=10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    model = Model(inputs=inputs, outputs=x)

    return model

# Now you can create the model
model = dl_model()
model.summary()