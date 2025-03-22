import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block: Splitting and depthwise separable convolutions
    def split_and_convolve(x):
        groups = tf.split(x, num_or_size_splits=3, axis=-1)
        
        conv_1x1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(groups[0])
        conv_3x3 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(groups[1])
        conv_5x5 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(groups[2])
        
        return Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    x = Lambda(split_and_convolve)(inputs)

    # Second block: Multiple branches for feature extraction
    branch_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)

    branch_2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    branch_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_2)

    branch_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    branch_3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_3)

    branch_4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_4 = Conv2D(32, (1, 1), padding='same', activation='relu')(branch_4)

    # Concatenate all branches
    x = Concatenate()([branch_1, branch_2, branch_3, branch_4])

    # Flatten and fully connected layer
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
model = dl_model()
model.summary()