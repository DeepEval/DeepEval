import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Construct the main path
    main_input = keras.Input(shape=input_shape, name='main_input')

    # Split the input into three groups
    group1, group2, group3 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(main_input)

    # Leave group1 unchanged
    group1 = group1

    # Perform feature extraction on group2 using a 3x3 convolution
    group2 = layers.Conv2D(32, (3, 3), strides=1, padding='same')(group2)
    group2 = layers.Activation('relu')(group2)

    # Combine group2 and group3 before passing through another 3x3 convolution
    combined = layers.Concatenate()([group2, group3])
    combined = layers.Conv2D(32, (3, 3), strides=1, padding='same')(combined)
    combined = layers.Activation('relu')(combined)

    # Concatenate the outputs of all three groups
    main_output = layers.Concatenate()([group1, combined])

    # Construct the branch path
    branch_input = keras.Input(shape=input_shape, name='branch_input')
    branch_output = layers.Conv2D(32, (1, 1), strides=1, padding='same')(branch_input)

    # Fuse the outputs from both the main and branch paths
    fused_output = layers.Add()([main_output, branch_output])

    # Flatten the output and pass it through a fully connected layer
    x = layers.Flatten()(fused_output)
    output = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=[main_input, branch_input], outputs=output)

    return model

# Build the model
model = dl_model()
model.summary()