import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense, Lambda, SeparableConv2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    def first_block(x):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
        
        # Flatten the outputs
        flat1 = Flatten()(pool1)
        flat2 = Flatten()(pool2)
        flat3 = Flatten()(pool3)
        
        # Apply dropout
        concat = Dropout(0.5)(tf.concat([flat1, flat2, flat3], axis=-1))
        
        return concat
    
    first_block_output = first_block(input_layer)
    
    # Reshape the output to a four-dimensional tensor
    reshape_layer = Lambda(lambda x: tf.reshape(x, (1, -1) + x.shape[1:]))(first_block_output)
    
    # Second block
    def second_block(x):
        # Split the input into four groups
        split1, split2, split3, split4 = tf.split(x, 4, axis=-1)
        
        # Process each group with separable convolutional layers
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split1)
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split2)
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split3)
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(split4)
        
        # Concatenate the outputs
        concat = Concatenate()([conv1, conv2, conv3, conv4])
        
        return concat
    
    second_block_output = second_block(reshape_layer)
    
    # Flatten the output
    flatten_layer = Flatten()(second_block_output)
    
    # Pass through a fully connected layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()