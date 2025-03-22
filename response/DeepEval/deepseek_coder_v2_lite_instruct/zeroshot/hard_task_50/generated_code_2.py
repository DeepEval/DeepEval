import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Lambda, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First block of the model
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(1, 1), strides=1)(x)
    x = Conv2D(64, (2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=4)(x)
    
    # Flatten the output of the pooling layers
    x_flat = Flatten()(x)
    x_dropout = Dropout(0.5)(x_flat)
    
    # Concatenate the flattened outputs
    x_concat = Concatenate()([x_flat, x_dropout])
    
    # Reshape the concatenated output into a four-dimensional tensor
    x_reshape = tf.reshape(x_concat, (-1, 1, 1, x_concat.shape[1]))
    
    # Second block of the model
    # Lambda layer to split the input into four groups along the last dimension
    x_split = Lambda(lambda tensor: tf.split(tensor, 4, axis=-1))(x_reshape)
    
    # Process each group with a separable convolutional layer
    conv_blocks = []
    for kernel_size in [(1, 1), (3, 3), (5, 5), (7, 7)]:
        conv = Conv2D(32, kernel_size, padding='same', activation='relu', depthwise_constraint=None)(x_split[0])
        conv = Conv2D(32, (1, 1), activation='relu')(conv)
        conv_blocks.append(conv)
    
    # Concatenate the outputs of the separable convolutional layers
    x_concat = Concatenate()(conv_blocks)
    
    # Flatten the output for the fully connected layer
    x_flatten = Flatten()(x_concat)
    
    # Fully connected layer for classification
    outputs = Dense(10, activation='softmax')(x_flatten)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model