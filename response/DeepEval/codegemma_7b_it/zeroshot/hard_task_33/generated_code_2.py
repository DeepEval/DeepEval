from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, BatchNormalization, Activation, Flatten, Dense, Concatenate

def dl_model():

    # Input layer
    input_img = Input(shape=(28, 28, 1))

    # Define the block function
    def block(x):
        conv_1x1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(x)
        dw_conv_3x3 = DepthwiseConv2D(kernel_size=3, strides=1, padding='same')(conv_1x1)
        conv_1x1_proj = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(dw_conv_3x3)
        add_layer = Add()([conv_1x1_proj, x])
        bn_layer = BatchNormalization()(add_layer)
        act_layer = Activation('relu')(bn_layer)
        return act_layer

    # Branch 1
    branch_1 = block(input_img)

    # Branch 2
    branch_2 = block(input_img)

    # Branch 3
    branch_3 = block(input_img)

    # Concatenate the outputs from the three branches
    concat_layer = Concatenate()([branch_1, branch_2, branch_3])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(concat_layer)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_img, outputs=dense_layer)

    return model