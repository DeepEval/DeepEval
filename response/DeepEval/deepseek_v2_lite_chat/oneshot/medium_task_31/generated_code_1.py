import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    split2 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    split3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Apply different convolutional kernels to each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split3[2])
    
    # Concatenate the outputs of the three groups
    concatenated = Concatenate()(outputs=[conv1, conv2, conv3])
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2))(concatenated)
    
    # Flatten and fully connected layers
    flatten = Flatten()(pool1)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model