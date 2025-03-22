import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def first_block(input_tensor):
        # Split the input into three groups
        split_1 = Lambda(lambda x: x[:, :16, :, :])(input_tensor)
        split_2 = Lambda(lambda x: x[:, 16:32, :, :])(input_tensor)
        split_3 = Lambda(lambda x: x[:, 32:, :, :])(input_tensor)
        
        # Depthwise separable convolutions
        depthwise_1x1 = Conv2D(64, (1, 1), padding='same', activation='relu')(split_1)
        depthwise_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu', depthwise_constraint=tf.keras.constraints.max_norm(1.))(split_2)
        depthwise_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu', depthwise_constraint=tf.keras.constraints.max_norm(1.))(split_3)
        
        # Batch normalization
        depthwise_1x1 = BatchNormalization()(depthwise_1x1)
        depthwise_3x3 = BatchNormalization()(depthwise_3x3)
        depthwise_5x5 = BatchNormalization()(depthwise_5x5)
        
        # Concatenate the outputs
        concatenated = Concatenate()([depthwise_1x1, depthwise_3x3, depthwise_5x5])
        return concatenated

    block_output = first_block(input_layer)

    # Second block
    def second_block(input_tensor):
        # First branch
        branch1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)
        
        # Second branch
        branch2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(64, (1, 7), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(64, (7, 1), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
        
        # Third branch
        branch3 = AveragePooling2D((3, 3), strides=1, padding='same')(input_tensor)
        branch3 = Conv2D(64, (1, 1), padding='same', activation='relu')(branch3)
        
        # Concatenate the outputs
        concatenated = Concatenate()([branch1, branch2, branch3])
        return concatenated

    second_block_output = second_block(block_output)

    # Flatten and fully connected layers
    flatten_layer = Flatten()(second_block_output)
    dense1 = Dense(256, activation='relu')(flatten_layer)
    output_layer = Dense(10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Create the model
model = dl_model()
model.summary()