import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Lambda, DepthwiseConv2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Flatten each pooled output and concatenate
    concat_layer = Concatenate()([Flatten()(pool1), Flatten()(pool2), Flatten()(pool3)])
    
    # Fully connected layer and reshape to 4D tensor
    fc_layer = Dense(units=256, activation='relu')(concat_layer)
    reshape_layer = Lambda(lambda x: tf.reshape(x, (-1, 2, 2, 64)))(fc_layer)
    
    # Second block
    def second_block(input_tensor):
        # Split the input into four groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(input_tensor)
        
        # Process each group with depthwise separable convolutional layers
        outputs = []
        for i, group in enumerate(split_layer):
            if i == 0:
                conv_layer = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(group)
            elif i == 1:
                conv_layer = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(group)
            elif i == 2:
                conv_layer = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(group)
            elif i == 3:
                conv_layer = DepthwiseConv2D(kernel_size=(7, 7), padding='same', activation='relu')(group)
            outputs.append(conv_layer)
        
        # Concatenate the outputs from all groups
        concat_output = Concatenate()(outputs)
        
        return concat_output
    
    second_block_output = second_block(reshape_layer)
    flatten_layer = Flatten()(second_block_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model