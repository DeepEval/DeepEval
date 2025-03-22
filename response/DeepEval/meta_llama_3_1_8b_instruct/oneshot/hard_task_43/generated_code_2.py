import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):
        # Block 1: Three parallel paths with average pooling layers of different scales
        avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=1)(input_tensor)
        avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)
        avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=4)(input_tensor)

        # Flatten the outputs of the pooling operations and concatenate them
        flatten1 = Flatten()(avgpool1)
        flatten2 = Flatten()(avgpool2)
        flatten3 = Flatten()(avgpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])

        return output_tensor
    
    block1_output = block1(input_layer)

    # Fully connected layer with ReLU activation
    dense = Dense(units=128, activation='relu')(block1_output)
    # Reshape the output to 4-dimensional tensor (batch_size, 1, 1, 128)
    reshape_layer = Reshape((1, 1, 128))(dense)

    def block2(input_tensor):
        # Block 2: Three branches for feature extraction
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = AveragePooling2D(pool_size=(2, 2), strides=2)(input_tensor)

        # Concatenate the outputs from all branches
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4, conv5, maxpool])

        return output_tensor
        
    block2_output = block2(reshape_layer)

    # Batch normalization and flatten the output
    batch_norm = BatchNormalization()(block2_output)
    flatten_layer = Flatten()(batch_norm)
    # Fully connected layer with ReLU activation
    dense2 = Dense(units=128, activation='relu')(flatten_layer)
    # Final fully connected layer with softmax activation for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model