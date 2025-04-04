import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, SeparableConv2D, Lambda, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Main path
        conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dropout_main = Dropout(0.2)(conv_main)
        conv_main_restore = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout_main)
        
        # Branch path
        conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Addition
        added = Add()([conv_main_restore, conv_branch])
        return added

    def block_2(input_tensor):
        # Split the input into three groups
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with separable convolutions
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layers[0])
        dropout1 = Dropout(0.2)(conv1)
        
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layers[1])
        dropout2 = Dropout(0.2)(conv2)
        
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layers[2])
        dropout3 = Dropout(0.2)(conv3)
        
        # Concatenate the outputs
        concatenated = Add()([dropout1, dropout2, dropout3])
        return concatenated

    # Apply the blocks to the input layer
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    # Flatten the output and pass it through a fully connected layer
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model