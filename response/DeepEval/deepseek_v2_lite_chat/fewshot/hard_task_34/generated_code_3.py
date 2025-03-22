import keras
from keras.layers import Input, Conv2D, ReLU, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=ReLU)(input_tensor)
        block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
        concat_block1 = Add()([conv1, block1])

        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat_block1)
        block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv2)
        concat_block2 = Add()([conv2, block2])

        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(concat_block2)
        block3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv3)
        concat_block3 = Add()([conv3, block3])

        return keras.Model(inputs=input_layer, outputs=concat_block3)

    def block_branch_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation=ReLU)(input_tensor)
        return keras.Model(inputs=input_tensor, outputs=conv1)

    main_path = block_main_path(input_layer)
    branch_path = block_branch_path(input_layer)
    fused_model = Add()([main_path.output, branch_path.output])
    flattened_fused_model = Flatten()(fused_model)
    output_layer = Dense(units=10, activation='softmax')(flattened_fused_model)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Construct and print the model
model = dl_model()
model.summary()