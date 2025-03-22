import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def first_block(input_tensor):
        # Split the input into three groups along the channel axis
        splits = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
        
        # Apply separable convolutions on each group
        outputs = []
        for i, split in enumerate(splits):
            if i == 0:  # 1x1 convolution
                conv = Conv2D(32, (1, 1), padding='same', activation='relu')(split)
            elif i == 1:  # 3x3 convolution
                conv = Conv2D(32, (3, 3), padding='same', activation='relu')(split)
            elif i == 2:  # 5x5 convolution
                conv = Conv2D(32, (5, 5), padding='same', activation='relu')(split)
            outputs.append(conv)
        
        # Concatenate the outputs from the three groups
        concatenated = Concatenate()(outputs)
        return concatenated

    block1_output = first_block(input_layer)
    batch_norm1 = BatchNormalization()(block1_output)

    # Second Block
    def second_block(input_tensor):
        # 3x3 convolution branch
        conv3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
        
        # 1x1 -> 3x3 -> 3x3 convolution branch
        conv1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1x1)
        conv3x3_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv3x3_1)
        
        # Max pooling branch
        max_pool = MaxPooling2D((3, 3), strides=1, padding='same')(input_tensor)
        
        # Concatenate the outputs from all branches
        concatenated = Concatenate()([conv3x3, conv3x3_2, max_pool])
        return concatenated

    block2_output = second_block(batch_norm1)
    batch_norm2 = BatchNormalization()(block2_output)

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(batch_norm2)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(gap)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()