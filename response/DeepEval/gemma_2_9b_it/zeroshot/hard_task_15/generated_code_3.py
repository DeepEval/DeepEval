import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Add

def dl_model():
    
    # Define input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    x_main = inputs
    x_main = Conv2D(32, (3, 3), activation='relu')(x_main)
    x_main = MaxPooling2D((2, 2))(x_main)
    x_main = Conv2D(64, (3, 3), activation='relu')(x_main)
    x_main = MaxPooling2D((2, 2))(x_main)
    x_main = GlobalAveragePooling2D()(x_main)
    
    # Fully connected layers for main path
    x_main = Dense(32, activation='relu')(x_main)
    x_main = Dense(3 * 32 * 32, activation='linear')(x_main)  # Output shape matches input
    x_main = Reshape((32, 32, 3))(x_main)

    # Branch path
    x_branch = inputs
    
    # Add main and branch outputs
    x = Add()([x_main, x_branch])

    # Fully connected layers for final classification
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model