from tensorflow.keras.layers import Input, Conv2D, Add, Dense, Flatten, Lambda, Reshape, Permute
from tensorflow.keras.layers import DepthwiseConv2D, SeparableConv2D, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10

    inputs = Input(shape=input_shape)

    # Direct branch to the input
    direct_branch = inputs
    
    # Block 1
    def block1(x):
        # Split the input into 3 groups
        groups = 3
        channels_per_group = x.shape[-1] // groups
        
        # Split into three groups
        split_groups = Lambda(lambda z: tf.split(z, num_or_size_splits=groups, axis=-1))(x)
        
        # Process each group with a 1x1 convolution
        processed_groups = [Conv2D(channels_per_group, (1, 1), padding='same', activation='relu')(sg) for sg in split_groups]
        
        # Concatenate the processed groups
        return Concatenate(axis=-1)(processed_groups)

    # Block 2
    def block2(x):
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        groups = 3
        channels_per_group = channels // groups
        
        # Reshape
        reshaped = Reshape((height, width, groups, channels_per_group))(x)
        
        # Permute dimensions to shuffle channels
        shuffled = Permute((1, 2, 4, 3))(reshaped)
        
        # Reshape back to original shape
        return Reshape((height, width, channels))(shuffled)

    # Block 3
    def block3(x):
        # 3x3 Depthwise Separable Convolution
        return SeparableConv2D(x.shape[-1], (3, 3), padding='same', activation='relu')(x)

    # Apply Block 1
    x = block1(inputs)
    
    # Apply Block 2
    x = block2(x)
    
    # Apply Block 3
    x = block3(x)
    
    # Apply Block 1 again
    x = block1(x)

    # Add the direct branch and main path
    x = Add()([x, direct_branch])

    # Global Average Pooling and Fully Connected Layer
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

model = dl_model()
model.summary()