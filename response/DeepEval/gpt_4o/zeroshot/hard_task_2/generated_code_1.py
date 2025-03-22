import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have shape 32x32x3

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Process each group through the series of convolutions
    conv_outputs = []
    for group in split_layer:
        x = Conv2D(32, (1, 1), activation='relu', padding='same')(group)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
        conv_outputs.append(x)

    # Combine the outputs of the three groups
    main_path = Add()(conv_outputs)

    # Fuse the main path with the original input
    fused = Add()([main_path, input_layer])

    # Flatten and add a fully connected layer for classification
    x = Flatten()(fused)
    output_layer = Dense(10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()