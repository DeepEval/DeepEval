import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Dense, Flatten, Concatenate
from tensorflow.keras import backend as K

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # First block
    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    main_path = Conv2D(3, (1, 1), activation='relu')(x)
    
    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Combine both paths
    x = Add()([main_path, branch_path])
    
    # Second block
    # Split the input into three groups along the channel
    split_points = [1, 2]
    split_layer = Lambda(lambda tensor: tf.split(tensor, num_or_size_splits=split_points, axis=3))
    split_groups = split_layer(x)
    
    # Extract features using depthwise separable convolutional layers with different kernel sizes
    def depthwise_separable_conv(x, kernel_size):
        pointwise_conv = Conv2D(256, (1, 1), activation='relu')(x)
        depthwise_conv = Conv2D(256, kernel_size, padding='same', group_size=256)(pointwise_conv)
        return depthwise_conv
    
    group1 = depthwise_separable_conv(split_groups[0], (1, 1))
    group2 = depthwise_separable_conv(split_groups[1], (3, 3))
    group3 = depthwise_separable_conv(split_groups[2], (5, 5))
    
    # Concatenate the outputs from the three groups
    x = Concatenate()([group1, group2, group3])
    
    # Flatten the output and add two fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()