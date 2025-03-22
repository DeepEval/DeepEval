import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Concatenate, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    # Input layer for CIFAR-10 images
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)

    # First block: splitting the input and applying depthwise separable convolutions
    def split_and_depthwise_convolutions(x):
        # Splitting the input tensor into 3 parts along the last dimension
        splits = tf.split(x, num_or_size_splits=3, axis=-1)
        outputs = []
        
        for i, split in enumerate(splits):
            kernel_size = (1 + 2 * i, 1 + 2 * i)  # Generates (1,1), (3,3), (5,5)
            x = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation=None, use_bias=False)(split)
            x = BatchNormalization()(x)
            x = tf.nn.relu(x)
            outputs.append(x)
        
        return Concatenate(axis=-1)(outputs)

    # Applying the first block
    x = Lambda(split_and_depthwise_convolutions)(inputs)

    # Second block: multiple branches for feature extraction
    branch1 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    branch1 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(branch1)

    branch2 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(x)
    branch2 = Conv2D(32, kernel_size=(1, 7), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(32, kernel_size=(7, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)

    branch3 = GlobalAveragePooling2D()(x)
    branch3 = tf.expand_dims(branch3, axis=1)  # Expand dimensions to match other branches
    branch3 = tf.expand_dims(branch3, axis=1)  # Expand dimensions to match other branches
    branch3 = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate outputs from all branches
    x = Concatenate(axis=-1)([branch1, branch2, branch3])

    # Fully connected layers for classification
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = Model(inputs, x)

    return model

# You can create the model and summarize it
model = dl_model()
model.summary()