import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Flatten, Concatenate, Dense, Reshape, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Average Pooling
    def block_1(input_tensor):
        maxpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        maxpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        maxpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    # Block 2: Parallel Paths for Multi-scale Feature Extraction
    def block_2(input_tensor):
        # Split input into four groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)

        # Path 1: 1x1 convolution, Path 2: 3x3 convolution, Path 3: 3x3 convolution, Path 4: 1x1 max pooling
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])

        # Dropout to mitigate overfitting
        conv1 = Dropout(0.2)(conv1)
        conv2 = Dropout(0.2)(conv2)
        conv3 = Dropout(0.2)(conv3)
        conv4 = Dropout(0.2)(conv4)

        # Concatenate outputs from each path
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        return output_tensor

    # Apply Block 1
    block1_output = block_1(input_tensor=input_layer)
    block1_output = Dense(units=64, activation='relu')(block1_output)
    block1_output = Reshape(target_shape=(4, 4, 4))(block1_output)

    # Apply Block 2
    block2_output = block_2(input_tensor=block1_output)

    # Flatten and fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()