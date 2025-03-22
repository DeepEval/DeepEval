import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Concatenate, Dense, Flatten, Add
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    splits = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Convolutional layers for each split
    conv1 = Conv2D(32, (1, 1), activation='relu', padding='same')(splits[0])
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(splits[1])
    conv3 = Conv2D(32, (5, 5), activation='relu', padding='same')(splits[2])

    # Concatenating the outputs of the convolutional layers
    main_path_output = Concatenate()([conv1, conv2, conv3])

    # Branch path
    branch_path_output = Conv2D(32, (1, 1), activation='relu', padding='same')(input_layer)

    # Combining main and branch paths through addition
    fused_features = Add()([main_path_output, branch_path_output])

    # Flatten and fully connected layers for classification
    flatten = Flatten()(fused_features)
    dense1 = Dense(128, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example of how to create the model
model = dl_model()
model.summary()