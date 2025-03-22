import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Split the input into three groups along the channel dimension
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)

    def process_group(group):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Process each group through a series of convolutions
    group1_output = process_group(inputs_groups[0])
    group2_output = process_group(inputs_groups[1])
    group3_output = process_group(inputs_groups[2])

    # Combine the outputs from the three groups
    combined_path = Add()([group1_output, group2_output, group3_output])

    # Fuse the combined path with the original input
    fused_output = Add()([combined_path, input_layer])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model