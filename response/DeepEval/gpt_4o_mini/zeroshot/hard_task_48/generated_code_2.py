import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Block 1
    # Split the input into three groups
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    conv_layers = []
    for i, kernel_size in enumerate([(1, 1), (3, 3), (5, 5)]):
        conv = layers.SeparableConv2D(filters=32, kernel_size=kernel_size, padding='same')(split_inputs[i])
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv_layers.append(conv)

    # Concatenate outputs of the three groups
    block1_output = layers.Concatenate()(conv_layers)

    # Block 2
    path1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(block1_output)

    path2 = layers.AveragePooling2D(pool_size=(3, 3), padding='same')(block1_output)
    path2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(path2)

    path3_conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(block1_output)
    path3_split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(path3_conv1)
    path3_conv2 = layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same')(path3_split[0])
    path3_conv3 = layers.Conv2D(filters=32, kernel_size=(3, 1), padding='same')(path3_split[1])
    path3 = layers.Concatenate()([path3_conv2, path3_conv3])

    path4_conv1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(block1_output)
    path4_conv2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path4_conv1)
    path4_split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(path4_conv2)
    path4_conv3 = layers.Conv2D(filters=32, kernel_size=(1, 3), padding='same')(path4_split[0])
    path4_conv4 = layers.Conv2D(filters=32, kernel_size=(3, 1), padding='same')(path4_split[1])
    path4 = layers.Concatenate()([path4_conv3, path4_conv4])

    # Concatenate outputs of the four paths
    block2_output = layers.Concatenate()([path1, path2, path3, path4])

    # Final classification layers
    flatten = layers.Flatten()(block2_output)
    output = layers.Dense(units=10, activation='softmax')(flatten)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=output)

    return model

# Example of creating the model
model = dl_model()
model.summary()