import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, SeparableConv2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10

    # Input layer
    inputs = Input(shape=input_shape)

    # Block 1: Multi-scale pooling
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    pool1_flat = Flatten()(pool1)

    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(inputs)
    pool2_flat = Flatten()(pool2)

    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(inputs)
    pool3_flat = Flatten()(pool3)

    # Concatenate flattened pools
    concatenated_pools = Concatenate()([pool1_flat, pool2_flat, pool3_flat])

    # Dropout for regularization
    dropout = Dropout(0.5)(concatenated_pools)

    # Fully connected layer
    fc = Dense(512, activation='relu')(dropout)

    # Reshape to 4D tensor for block 2 processing
    reshape = Reshape((8, 8, 8))(fc)

    # Block 2: Split and SeparableConv2D
    def split_and_process(x):
        # Splitting into four groups along the last dimension
        splits = tf.split(x, num_or_size_splits=4, axis=-1)

        # Separable Convolutions with different kernel sizes
        conv1 = SeparableConv2D(32, kernel_size=(1, 1), activation='relu', padding='same')(splits[0])
        conv2 = SeparableConv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(splits[1])
        conv3 = SeparableConv2D(32, kernel_size=(5, 5), activation='relu', padding='same')(splits[2])
        conv4 = SeparableConv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(splits[3])

        # Concatenate the outputs
        return Concatenate()([conv1, conv2, conv3, conv4])

    # Lambda layer to use split_and_process function
    processed = Lambda(split_and_process)(reshape)

    # Flatten and output
    flat = Flatten()(processed)
    outputs = Dense(num_classes, activation='softmax')(flat)

    # Construct model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of compiling and summarizing the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()