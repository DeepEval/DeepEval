import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Add, Concatenate, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Permute, Reshape, Flatten, Dense

def dl_model():
    def block1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        conv3 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv2)
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        branch_output = Add()[(2, 3)]([conv2, conv4])
        return branch_output

    def block2(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        input_tensor = Reshape((shape[1] * shape[2], shape[3]))(input_tensor)
        groups = 4
        channels_per_group = shape[3] // groups
        input_tensor = Permute((2, 3, 1))(input_tensor)
        reshaped_tensor = Reshape((shape[1], shape[2], groups, channels_per_group))(input_tensor)
        permuted_tensor = Permute((3, 2, 1))(reshaped_tensor)
        output_tensor = Flatten()(permuted_tensor)
        dense1 = Dense(units=128, activation='relu')(output_tensor)
        output_layer = Dense(units=10, activation='softmax')(dense1)
        return output_layer

    input_layer = Input(shape=(28, 28, 1))
    block1_output = block1(input_layer)
    batch_norm1 = BatchNormalization()(block1_output)
    activation1 = Activation('relu')(batch_norm1)

    block2_output = block2(activation1)
    model = keras.Model(inputs=input_layer, outputs=block2_output)

    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])