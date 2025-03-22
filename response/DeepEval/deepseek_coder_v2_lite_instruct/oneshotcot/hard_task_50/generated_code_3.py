import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, Concatenate, Dropout, Dense, Flatten, Lambda, Conv2D, SeparableConv2D
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    def first_block(input_tensor):
        # Max pooling layers with different scales
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        # Flatten each pooling output
        flatten1 = Flatten()(pool1)
        flatten2 = Flatten()(pool2)
        flatten3 = Flatten()(pool3)
        
        # Dropout to mitigate overfitting
        dropout = Dropout(0.5)(Concatenate()([flatten1, flatten2, flatten3]))
        
        # Transform the output into a four-dimensional tensor
        reshape = Lambda(lambda x: tf.expand_dims(x, -1))(dropout)
        reshape = Lambda(lambda x: tf.expand_dims(x, -1))(reshape)
        
        return reshape
    
    reshape_output = first_block(input_layer)
    
    # Second block
    def second_block(input_tensor):
        # Split the input into four groups
        split_layer = Lambda(lambda x: tf.split(x, 4, axis=-1))(input_tensor)
        
        # Process each group with a separable convolutional layer
        conv_groups = []
        for group in split_layer:
            conv1x1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(group)
            conv3x3 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(group)
            conv5x5 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(group)
            conv7x7 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(group)
            conv_groups.append(conv1x1)
            conv_groups.append(conv3x3)
            conv_groups.append(conv5x5)
            conv_groups.append(conv7x7)
        
        # Concatenate the outputs
        concat_output = Concatenate()(conv_groups)
        
        return concat_output
    
    concat_output = second_block(reshape_output)
    
    # Flatten the output and pass through a fully connected layer
    flatten_layer = Flatten()(concat_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model