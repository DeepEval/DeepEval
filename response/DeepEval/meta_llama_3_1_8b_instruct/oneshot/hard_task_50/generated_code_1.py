import keras
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dropout, Dense, Reshape, Lambda
from keras.layers import SeparableConv2D
from keras.backend import tf as ktf
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First block: process the input through three max pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(pool2)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(pool3)
    
    # Concatenate the flattened outputs with dropout to prevent overfitting
    concat_output = Concatenate()([flatten1, flatten2, flatten3])
    dropout_output = Dropout(0.2)(concat_output)
    
    # Reshape the output for processing in the second block
    reshape_output = Reshape((1024,))(dropout_output)
    
    # Second block: split the input into four groups and process each group separately
    split_layer = Lambda(lambda x: ktf.split(x, num_or_size_splits=4, axis=1))(reshape_output)
    
    conv_output1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv_output2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer[1])
    conv_output3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(split_layer[2])
    conv_output4 = SeparableConv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same')(split_layer[3])
    
    # Concatenate the outputs from the four groups
    concat_output2 = Concatenate()([conv_output1, conv_output2, conv_output3, conv_output4])
    
    # Flatten the output and pass it through a fully connected layer to produce the final classification
    flatten_layer = Flatten()(concat_output2)
    dense_output = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model