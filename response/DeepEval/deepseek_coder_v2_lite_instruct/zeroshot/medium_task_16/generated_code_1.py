import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Dense, Flatten

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Apply 1x1 convolutions to each group independently
    conv_groups = []
    for i, split in enumerate(splits):
        conv = Conv2D(filters=split.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(split)
        conv_groups.append(conv)
    
    # Downsample each group via average pooling
    pooled_groups = []
    for conv_group in conv_groups:
        pooled = AveragePooling2D(pool_size=(8, 8), strides=(8, 8))(conv_group)
        pooled_groups.append(pooled)
    
    # Concatenate the resulting groups of feature maps along the channel dimension
    concatenated = tf.concat(pooled_groups, axis=-1)
    
    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened = Flatten()(concatenated)
    
    # Pass through two fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(fc1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()