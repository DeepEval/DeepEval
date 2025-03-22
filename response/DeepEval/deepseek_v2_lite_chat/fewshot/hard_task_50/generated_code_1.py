import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Assuming CIFAR-10 images are 32x32x3

    def block_1(input_tensor):
        # First max pooling layer
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        # Second max pooling layer
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        # Third max pooling layer
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        # Flatten the pooled outputs
        flatten1 = Flatten()(maxpool1)
        flatten2 = Flatten()(maxpool2)
        flatten3 = Flatten()(maxpool3)
        # Dropout to prevent overfitting
        dropout = Dropout(0.5)(flatten1)
        dropout = Dropout(0.5)(flatten2)
        dropout = Dropout(0.5)(flatten3)
        # Concatenate the flattened pooled outputs
        concat = Concatenate()(dropout)
        return concat

    def block_2(input_tensor):
        # Split input into four groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        # Process each group with a separable convolution
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv4 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
        # Concatenate the outputs of the separably convolved groups
        concat = Concatenate()([conv1, conv2, conv3, conv4])
        return concat

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

dl_model = dl_model()