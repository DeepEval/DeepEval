import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input Layer
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Block 1: Splitting the input and applying separable convolutions
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Separable Convolutions with different kernel sizes
    conv1 = layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_inputs[0])
    conv2 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_inputs[1])
    conv3 = layers.SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_inputs[2])
    
    # Concatenating the outputs from the three branches
    block1_output = layers.Concatenate()([conv1, conv2, conv3])
    
    # Block 2: Multiple branches for enhanced feature extraction
    # Branch 1: 3x3 Convolution
    branch1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(block1_output)
    
    # Branch 2: 1x1 Conv followed by two 3x3 Convolutions
    branch2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(block1_output)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    
    # Branch 3: Max Pooling
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(block1_output)
    
    # Concatenating the outputs from all branches
    block2_output = layers.Concatenate()([branch1, branch2, branch3])
    
    # Global Average Pooling
    global_avg_pool = layers.GlobalAveragePooling2D()(block2_output)
    
    # Fully Connected Layer for classification
    dense_output = layers.Dense(10, activation='softmax')(global_avg_pool)

    # Creating the model
    model = models.Model(inputs=inputs, outputs=dense_output)
    
    return model

# Example usage:
model = dl_model()
model.summary()