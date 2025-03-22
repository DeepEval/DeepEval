import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, Conv2D, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

def dl_model():
    # Load CIFAR-10 dataset to obtain the shape of the input
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = x_train.shape[1:]

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Split the input into three groups along the channel axis
    split_channels = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Multi-scale feature extraction on each group
    conv1 = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_channels[0])
    conv2 = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_channels[1])
    conv3 = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_channels[2])

    # Concatenate the outputs of the separable convolutions
    main_path = Concatenate(axis=-1)([conv1, conv2, conv3])

    # Branch path with a 1x1 convolution
    branch_path = Conv2D(96, (1, 1), padding='same', activation='relu')(inputs)

    # Fuse the main and branch paths through addition
    fused = Add()([main_path, branch_path])

    # Flatten the combined output
    flat = Flatten()(fused)

    # Fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flat)
    outputs = Dense(10, activation='softmax')(fc1)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()

# Print model summary
model.summary()