import keras
from keras.layers import Input, Conv2D, Dropout, Concatenate, Flatten, Dense, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_input[0])
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_input[1])
    conv_5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_input[2])
    
    # Apply dropout to reduce overfitting
    dropout_layer = Dropout(rate=0.5)(Concatenate()([conv_1x1, conv_3x3, conv_5x5]))

    # Block 2
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout_layer)
    
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(dropout_layer)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(branch3)
    
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(dropout_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch4)
    
    # Concatenate the outputs of all branches
    block2_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model