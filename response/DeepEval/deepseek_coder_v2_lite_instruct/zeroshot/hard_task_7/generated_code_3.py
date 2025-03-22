import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Concatenate, Reshape, Permute, Dense, Flatten

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolutional layer
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    
    # Block 1
    # Split the input into two groups
    split_index = tf.shape(x)[-1] // 2
    split_layer = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=2, axis=-1))
    x1, x2 = split_layer(x)
    
    # Group 1 operations
    y1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(x1)
    y1 = Conv2D(32, kernel_size=(3, 3), groups=32, activation='relu')(y1)
    y1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(y1)
    
    # Group 2 operations (pass through)
    y2 = x2
    
    # Merge the outputs
    x = Concatenate()([y1, y2])
    
    # Block 2
    # Get the shape of the input
    shape = tf.shape(x)
    
    # Reshape the input into four groups
    groups = 4
    height, width, channels = shape[1], shape[2], shape[3]
    channels_per_group = channels // groups
    reshaped = Reshape((height, width, groups, channels_per_group))(x)
    
    # Swap the third and fourth dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to the original shape to achieve channel shuffling
    final_shape = (height, width, channels)
    x = Reshape(final_shape)(permuted)
    
    # Flatten the final output
    x = Flatten()(x)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()