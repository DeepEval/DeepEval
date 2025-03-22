import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Reshape, Permute, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)
    x = tf.split(x, num_or_size_splits=3, axis=-1)
    x = [Conv2D(32, (1, 1), activation='relu')(xi) for xi in x]
    x = Concatenate(axis=-1)(x)
    
    # Block 2
    shape = tf.keras.backend.int_shape(x)
    height, width, channels = shape[1], shape[2], shape[3]
    x = Reshape((height, width, 3, int(channels / 3))(x)
    x = Permute((1, 2, 4, 3))(x)
    x = Reshape((height, width, channels))(x)
    
    # Block 3
    x = Conv2D(64, (3, 3), depthwise_constraint=True, activation='relu')(x)
    
    # Branch path
    branch = AveragePooling2D((4, 4))(x)
    branch = Flatten()(branch)
    
    # Concatenate main path and branch path
    combined = Concatenate(axis=-1)([x, branch])
    
    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(combined)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()