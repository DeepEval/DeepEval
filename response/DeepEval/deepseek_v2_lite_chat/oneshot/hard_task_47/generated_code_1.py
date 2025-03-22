import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input shape should match the size of the CIFAR-10 images (32x32x3)
    input_layer = Input(shape=(32, 32, 3))

    # First block for feature extraction
    def feature_extraction(input_tensor):
        # Split the input into three groups along the last dimension
        split1, split2, split3 = tf.split(input_tensor, num_or_size_splits=3, axis=-1)

        # Depthwise separable convolutional layers
        conv1 = SeparableConv2D(filters=32, kernel_size=3, activation='relu')(split1)
        conv2 = SeparableConv2D(filters=32, kernel_size=3, activation='relu')(split2)
        conv3 = SeparableConv2D(filters=32, kernel_size=3, activation='relu')(split3)

        # Batch normalization and MaxPooling for feature extraction
        batch_norm1 = BatchNormalization()(conv1)
        batch_norm2 = BatchNormalization()(conv2)
        batch_norm3 = BatchNormalization()(conv3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(batch_norm2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(batch_norm3)

        # Concatenate the outputs from the three groups
        concatenated = Concatenate()(outputs=[pool1, pool2, pool3])
        return concatenated

    # Second block for feature extraction with multiple branches
    def multiple_branch_extraction(input_tensor):
        # Multiple branches for feature extraction
        branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        branch3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_tensor)
        branch4 = Conv2D(filters=64, kernel_size=(1, 7), padding='valid', activation='relu')(input_tensor)
        branch5 = Conv2D(filters=64, kernel_size=(7, 1), padding='valid', activation='relu')(input_tensor)
        branch6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)

        # Average pooling for each branch
        pool1 = MaxPooling2D(pool_size=(4, 4))(branch1)
        pool2 = MaxPooling2D(pool_size=(4, 4))(branch2)
        pool3 = MaxPooling2D(pool_size=(4, 4))(branch3)
        pool4 = MaxPooling2D(pool_size=(4, 4))(branch4)
        pool5 = MaxPooling2D(pool_size=(4, 4))(branch5)
        pool6 = MaxPooling2D(pool_size=(4, 4))(branch6)

        # Concatenate the outputs from all branches
        concatenated = Concatenate()(outputs=[pool1, pool2, pool3, pool4, pool5, pool6])
        return concatenated

    # Apply the first block
    first_block_output = feature_extraction(input_layer)

    # Apply the second block
    second_block_output = multiple_branch_extraction(first_block_output)

    # Flatten and add fully connected layers
    flatten = Flatten()(second_block_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()