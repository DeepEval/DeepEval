import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Dense, Lambda
from tensorflow.keras.optimizers import Adam


def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Data augmentation
    datagen = ImageDataGenerator(rotation=45, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(x_train)

    # Model input
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    block1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    block1 = [Conv2D(32, (1, 1), activation='relu')(b) for b in block1]
    block1_features = tf.concat(block1, axis=-1)  # Concatenate along the channels axis

    # Block 2
    block2 = tf.reshape(block1_features, (block1_features.shape[0], block1_features.shape[1], block1_features.shape[2], 3))
    block2 = tf.transpose(block2, perm=[0, 3, 1, 2])  # Swap height and width, reshape to (batch_size, 3, height, width)
    block2 = tf.reshape(block2, (block2.shape[0], block2.shape[1], block2.shape[2]*block2.shape[3]))  # Flatten to (batch_size, 3, groups*channels_per_group)
    block2 = Conv2D(64, (3, 3), padding='same')(block2)  # Depthwise separable convolution

    # Block 3
    block3 = SeparableConv2D(64, (3, 3), padding='same')(block2)

    # Branch path: Average pooling
    branch_output = AveragePooling2D(pool_size=(3, 3))(inputs)

    # Concatenate features from main path and branch path
    concat_model = Concatenate()([block3, branch_output])

    # Add fully connected layers
    outputs = Dense(10, activation='softmax')(concat_model)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model