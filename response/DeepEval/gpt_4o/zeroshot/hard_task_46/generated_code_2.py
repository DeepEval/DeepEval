import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define input shape for CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First block: Splitting and applying separable convolutions
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split = Lambda(split_channels)(inputs)
    
    sep_conv1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split[0])
    sep_conv3 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split[1])
    sep_conv5 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split[2])
    
    concatenated1 = Concatenate()([sep_conv1, sep_conv3, sep_conv5])
    
    # Second block: Multiple branches
    # Branch 1: Single 3x3 Convolution
    branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(concatenated1)
    
    # Branch 2: Series of 1x1 and two 3x3 Convolutions
    branch2 = Conv2D(64, (1, 1), padding='same', activation='relu')(concatenated1)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(concatenated1)
    
    # Concatenate all branches
    concatenated2 = Concatenate()([branch1, branch2, branch3])
    
    # Global average pooling
    gap = GlobalAveragePooling2D()(concatenated2)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(gap)
    
    # Define and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model