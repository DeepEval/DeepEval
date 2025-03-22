import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Flatten the outputs of the max pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Apply dropout to mitigate overfitting
    dropout = Dropout(0.5)(tf.concat([flatten1, flatten2, flatten3], axis=-1))
    
    # Transform the output into a 4D tensor
    reshape = Lambda(lambda x: tf.expand_dims(x, -1))(dropout)
    reshape = Lambda(lambda x: tf.expand_dims(x, -1))(reshape)
    
    # Second Block
    def second_block(input_tensor):
        # Split the input into four groups
        split1 = Lambda(lambda x: tf.split(x, 4, axis=-1)[0])(input_tensor)
        split2 = Lambda(lambda x: tf.split(x, 4, axis=-1)[1])(input_tensor)
        split3 = Lambda(lambda x: tf.split(x, 4, axis=-1)[2])(input_tensor)
        split4 = Lambda(lambda x: tf.split(x, 4, axis=-1)[3])(input_tensor)
        
        # Process each group with separable convolutions
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split1)
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split2)
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split3)
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split4)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])
        
        return output_tensor
    
    block_output = second_block(reshape)
    
    # Flatten the output and pass it through a fully connected layer
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model