import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, SeparableConv2D, GlobalAveragePooling2D, Dense, Add, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block1(x):
        # Split the input into three groups along the channel axis
        split_groups = Lambda(lambda z: tf.split(z, num_or_size_splits=3, axis=-1))(x)
        # Process each group with a 1x1 convolution, reducing channels to one-third
        conv_groups = [Conv2D(filters=x.shape[-1] // 3, kernel_size=1, activation='relu')(group) for group in split_groups]
        # Concatenate the groups along the channel axis
        return Concatenate(axis=-1)(conv_groups)

    block1_out = block1(input_layer)

    # Block 2
    def block2(x):
        # Get the shape of the feature
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        groups = 3
        channels_per_group = channels // groups
        # Reshape into (height, width, groups, channels_per_group)
        reshaped = tf.reshape(x, (-1, height, width, groups, channels_per_group))
        # Permute dimensions (swap groups and channels_per_group)
        permuted = tf.transpose(reshaped, perm=[0, 1, 2, 4, 3])
        # Reshape back to original shape
        reshaped_back = tf.reshape(permuted, (-1, height, width, channels))
        return reshaped_back

    block2_out = Lambda(block2)(block1_out)

    # Block 3
    block3_out = SeparableConv2D(filters=block2_out.shape[-1], kernel_size=(3, 3), padding='same', activation='relu')(block2_out)

    # Direct branch from input
    direct_branch = Conv2D(filters=block3_out.shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine main path and direct branch
    combined = Add()([block3_out, direct_branch])

    # Block 1 again on combined
    final_block1_out = block1(combined)

    # Global Average Pooling and Fully Connected Layer for classification
    global_avg_pooling = GlobalAveragePooling2D()(final_block1_out)
    output_layer = Dense(10, activation='softmax')(global_avg_pooling)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()