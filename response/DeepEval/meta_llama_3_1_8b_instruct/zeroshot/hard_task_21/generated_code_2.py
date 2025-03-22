# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Define the deep learning model function
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the main path
    main_path = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    main_path = layers.DepthwiseConv2D(kernel_size=1, name='main_path_conv_1x1')(main_path[0])
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.Activation('relu')(main_path)
    main_path = layers.DepthwiseConv2D(kernel_size=3, name='main_path_conv_3x3')(main_path)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.Activation('relu')(main_path)
    main_path = layers.DepthwiseConv2D(kernel_size=5, name='main_path_conv_5x5')(main_path)
    main_path = layers.BatchNormalization()(main_path)
    main_path = layers.Activation('relu')(main_path)
    main_path = layers.Concatenate()(main_path)

    # Define the branch path
    branch_path = layers.Conv2D(kernel_size=1, name='branch_path_conv_1x1')(inputs)
    branch_path = layers.BatchNormalization()(branch_path)
    branch_path = layers.Activation('relu')(branch_path)

    # Add the main and branch paths
    outputs = layers.Add()([main_path, branch_path])

    # Flatten the output
    outputs = layers.Flatten()(outputs)

    # Define the fully connected layers
    outputs = layers.Dense(128, activation='relu')(outputs)
    outputs = layers.Dropout(0.2)(outputs)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Construct the deep learning model
model = dl_model()
model.summary()