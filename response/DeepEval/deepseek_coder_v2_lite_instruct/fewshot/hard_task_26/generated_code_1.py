import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_initial = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_initial)
    
    # Branch 2
    branch2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv_initial)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Lambda(lambda x: tf.image.resize(x, (32, 32)))(branch2)
    
    # Branch 3
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv_initial)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Lambda(lambda x: tf.image.resize(x, (32, 32)))(branch3)
    
    # Concatenate outputs from all branches
    main_path = Concatenate()([branch1, branch2, branch3])
    
    # Final 1x1 convolutional layer
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add outputs from both paths
    added = Add()([main_path, branch_path])
    
    # Flatten and pass through two fully connected layers
    flatten_layer = Flatten()(added)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model