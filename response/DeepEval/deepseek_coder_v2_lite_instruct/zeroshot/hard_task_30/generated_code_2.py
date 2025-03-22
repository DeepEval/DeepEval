import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # First block: Dual-path structure
    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Conv2D(3, (3, 3), padding='same')(main_path)
    
    # Branch path
    branch_path = Conv2D(32, (1, 1), padding='same')(inputs)
    
    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Second block: Depthwise separable convolutional layers
    # Split the input into three groups
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(combined)
    
    # Extract features using depthwise separable convolutional layers with different kernel sizes
    def depthwise_separable_conv(x, kernel_size):
        x = Conv2D(x.shape[-1], kernel_size, padding='same', depthwise_constraint=None, pointwise_constraint=None)(x)
        x = Conv2D(x.shape[-1], (1, 1), activation='relu')(x)
        return x
    
    output_1x1 = depthwise_separable_conv(split_1[0], (1, 1))
    output_3x3 = depthwise_separable_conv(split_1[1], (3, 3))
    output_5x5 = depthwise_separable_conv(split_1[2], (5, 5))
    
    # Concatenate the outputs
    outputs = Add()([output_1x1, output_3x3, output_5x5])

    # Flatten the output
    flattened = Flatten()(outputs)

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model