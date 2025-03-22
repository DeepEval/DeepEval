import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Add, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32 images with 3 color channels

    # First block: Main path and branch path
    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        drop1 = Dropout(rate=0.2)(conv1)
        conv2 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same')(drop1)
        branch_path = input_tensor
        adding_layer = Add()([conv2, branch_path])
        return adding_layer

    block1_output = block_1(input_tensor=input_layer)

    # Second block: Three groups with separable convolutional layers
    def block_2(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Dense(32, activation='relu')(inputs_groups[0])
        conv1 = Dropout(rate=0.2)(conv1)
        conv1 = Dense(3, activation='relu')(conv1)

        conv2 = Dense(32, activation='relu')(inputs_groups[1])
        conv2 = Dropout(rate=0.2)(conv2)
        conv2 = Dense(3, activation='relu')(conv2)

        conv3 = Dense(32, activation='relu')(inputs_groups[2])
        conv3 = Dropout(rate=0.2)(conv3)
        conv3 = Dense(3, activation='relu')(conv3)

        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    block2_output = block_2(input_tensor=block1_output)

    # Flattening and final fully connected layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model