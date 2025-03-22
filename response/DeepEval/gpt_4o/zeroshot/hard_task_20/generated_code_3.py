import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 image dimensions
    num_classes = 10  # CIFAR-10 has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)

    # Main Path
    # Split input into three parts
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Apply convolutions of different kernel sizes
    conv1x1 = Conv2D(32, (1, 1), activation='relu', padding='same')(split_channels[0])
    conv3x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(split_channels[1])
    conv5x5 = Conv2D(32, (5, 5), activation='relu', padding='same')(split_channels[2])
    
    # Concatenate the results
    main_path_output = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])

    # Branch Path
    branch_path_output = Conv2D(96, (1, 1), activation='relu', padding='same')(inputs)

    # Fusion
    fused_output = Add()([main_path_output, branch_path_output])

    # Classification Layers
    x = Flatten()(fused_output)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=x)

    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()