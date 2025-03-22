import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, BatchNormalization, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # First Block
    def split_and_process(x):
        # Split input into three groups along the channel dimension
        splits = tf.split(x, num_or_size_splits=3, axis=-1)
        
        # First group with 1x1 depthwise separable convolution
        group1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same')(splits[0])
        group1 = BatchNormalization()(group1)
        
        # Second group with 3x3 depthwise separable convolution
        group2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(splits[1])
        group2 = BatchNormalization()(group2)
        
        # Third group with 5x5 depthwise separable convolution
        group3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same')(splits[2])
        group3 = BatchNormalization()(group3)
        
        # Concatenate outputs from all groups
        return tf.concat([group1, group2, group3], axis=-1)
    
    block1_output = Lambda(split_and_process)(input_layer)

    # Second Block
    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(block1_output)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)
    
    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch2 = Conv2D(64, (1, 1), activation='relu', padding='same')(block1_output)
    branch2 = Conv2D(64, (1, 7), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (7, 1), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    
    # Branch 3: Average pooling
    branch3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(block1_output)
    
    # Concatenate outputs from all branches
    block2_output = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Fully connected layers for classification
    flatten = Flatten()(block2_output)
    fc1 = Dense(256, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(fc1)  # CIFAR-10 has 10 classes

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()

# Display the model summary
model.summary()