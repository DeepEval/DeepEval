import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda, Reshape
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Flatten the outputs of the max pooling layers
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Dropout to mitigate overfitting
    dropout = Dropout(0.5)(Concatenate()([flatten1, flatten2, flatten3]))
    
    # Reshape the output to a four-dimensional tensor
    reshape = Reshape((3, 3, 3))(dropout)
    
    # Second block
    def separable_conv_block(inputs, kernel_size):
        return Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
    
    conv1 = separable_conv_block(reshape, (1, 1))
    conv2 = separable_conv_block(reshape, (3, 3))
    conv3 = separable_conv_block(reshape, (5, 5))
    conv4 = separable_conv_block(reshape, (7, 7))
    
    # Concatenate the outputs of the separable convolutional layers
    concat = Concatenate()([conv1, conv2, conv3, conv4])
    
    # Flatten the concatenated output
    flatten = Flatten()(concat)
    
    # Fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=dense)
    
    return model

# Create the model
model = dl_model()
model.summary()