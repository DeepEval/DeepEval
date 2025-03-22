import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, Lambda, Reshape, Permute, DepthwiseConv2D

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Conv2D(64, (1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Splitting the input into three groups
    x1, x2, x3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x)
    
    # Processing each group with a 1x1 convolutional layer
    x1 = Conv2D(21, (1, 1), padding='same')(x1)
    x2 = Conv2D(21, (1, 1), padding='same')(x2)
    x3 = Conv2D(21, (1, 1), padding='same')(x3)
    
    # Concatenating the outputs along the channel dimension
    x = tf.concat([x1, x2, x3], axis=-1)
    
    # Block 2
    # Obtaining the shape of the feature from Block 1
    shape = tf.keras.backend.int_shape(x)
    height, width, channels = shape[1], shape[2], shape[3]
    
    # Reshaping into groups
    x = Reshape((height, width, 3, channels // 3))(x)
    
    # Swapping the third and fourth dimensions
    x = Permute((1, 2, 4, 3))(x)
    
    # Reshaping back to the original shape
    x = Reshape((height, width, channels))(x)
    
    # Block 3
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Branch connecting directly to the input
    branch = Conv2D(64, (1, 1), padding='same')(inputs)
    branch = BatchNormalization()(branch)
    branch = Activation('relu')(branch)
    
    # Combining the outputs from the main path and the branch through addition
    x = Add()([x, branch])
    
    # Final output layer
    outputs = Conv2D(10, (1, 1), activation='softmax')(x)
    
    # Model construction
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()