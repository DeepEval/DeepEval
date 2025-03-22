import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Conv2D, Flatten, Concatenate, Average, Permute, Reshape
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Block 1
    x = inputs
    x = Lambda(lambda z: tf.split(z, 3, axis=-1))(inputs)
    for _ in range(3):
        x = Conv2D(32, (1, 1), activation='relu')(x)
    x = tf.concat(x, axis=-1)  # Concatenate the outputs from the three groups
    
    # Block 2
    shape = tf.shape(x)
    x = Permute((2, 3, 1))(x)  # Swap dimensions to (batch_size, groups, channels)
    x = Reshape((shape[1] // 3, shape[2], 3))(x)  # Reshape to (height, width, groups, channels_per_group)
    x = Conv2D(64, (3, 3), padding='same')(x)  # Additional 1x1 conv for reducing channels
    
    # Block 3
    x = DepthwiseConv2D((3, 3), kernel_initializer='he_normal')(x)  # Depthwise separable convolution
    x = Conv2D(64, (1, 1), activation='relu')(x)  # Additional 1x1 conv for depth dimension
    
    # Branch path
    avg_pool = Average()(x)  # Average pooling to extract features
    
    # Main path and branch path concatenation
    concat = Concatenate()([x, avg_pool])  # Concatenate the outputs of main path and branch path
    
    # Classification
    outputs = Dense(10, activation='softmax')(concat)  # Assuming 10 classes for CIFAR-10
    
    # Model construction
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()