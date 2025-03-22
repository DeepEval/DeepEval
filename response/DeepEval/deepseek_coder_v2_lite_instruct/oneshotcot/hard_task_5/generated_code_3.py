import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Permute, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def block1(x):
        # Split the input into three groups
        splits = [Lambda(lambda z: tf.split(z, 3, axis=-1))(x), Lambda(lambda z: tf.split(z, 3, axis=-1))(x), Lambda(lambda z: tf.split(z, 3, axis=-1))(x)]
        # Process each group with a 1x1 convolutional layer
        processed = [Conv2D(filters=x.shape[-1]//3, kernel_size=(1, 1), padding='same', activation='relu')(split) for split in splits]
        # Concatenate along the channel dimension
        return Concatenate(axis=-1)(processed)
    
    def block2(x):
        # Get the shape of the feature map
        shape = x.shape
        # Reshape into groups
        reshaped = Lambda(lambda z: tf.reshape(z, (shape[1], shape[2], 3, shape[3]//3)))(x)
        # Permute dimensions to swap the third and fourth dimensions
        permuted = Permute((1, 2, 4, 3))(reshaped)
        # Reshape back to original shape
        return Permute((1, 2, 4, 3))(permuted)
    
    def block3(x):
        # Process with 3x3 depthwise separable convolution
        return Conv2D(filters=x.shape[-1], kernel_size=(3, 3), padding='same', depthwise_mode=True, activation='relu')(x)
    
    # Apply blocks in sequence
    x = block1(input_layer)
    x = block2(x)
    x = block3(x)
    x = block1(input_layer)  # Block 1 again

    # Add branch from the input
    branch = Conv2D(filters=x.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main path and branch
    combined = Add()([x, branch])

    # Flatten and pass through fully connected layers
    flattened = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.summary()