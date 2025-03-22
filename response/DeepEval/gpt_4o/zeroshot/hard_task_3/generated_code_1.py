from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, Dense, Flatten, Lambda, Add, Concatenate
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    num_classes = 10  # CIFAR-10 dataset has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split input into three groups along the channel dimension
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    processed_splits = []
    for i in range(3):
        x = Conv2D(32, (1, 1), activation='relu')(split_channels[i])
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Dropout(0.5)(x)
        processed_splits.append(x)

    # Concatenate the outputs from the three groups
    main_pathway = Concatenate(axis=-1)(processed_splits)

    # Parallel branch pathway with a 1x1 convolution
    branch_pathway = Conv2D(main_pathway.shape[-1], (1, 1), activation='relu')(inputs)

    # Combine the main pathway and branch pathway using addition
    combined = Add()([main_pathway, branch_pathway])

    # Flatten and add fully connected layer for classification
    x = Flatten()(combined)
    x = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=x)

    return model

# Example usage:
# model = dl_model()
# model.summary()