import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Initial convolutional layer with 32 kernels
    conv1 = Conv2D(32, (3, 3), activation='sigmoid')(input_layer)
    
    # Block 1
    # Split the input into two groups
    split1 = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv1)
    
    # Operations for the first group
    conv2 = Conv2D(32, (1, 1), activation='relu')(split1[0])
    depthwise = Conv2D(32, (3, 3), activation='relu')(split1[1])
    conv3 = Conv2D(64, (1, 1), activation='relu')(depthwise)
    
    # Merge the outputs of the first group
    merged1 = Concatenate()([conv2, conv3])
    
    # Operations for the second group
    no_op = split1[1]  # No operation for the second group
    
    # Merge the outputs of the second group
    merged2 = no_op
    
    # Merge the outputs of both groups
    merged = Concatenate()([merged1, merged2])
    
    # Block 2
    # Split the merged output into four groups
    split2 = Lambda(lambda x: tf.split(x, 4, axis=-1))(merged)
    
    # Reshape and channel shuffle
    reshape = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[3], tf.shape(x)[2], tf.shape(x)[4])))(split2[0])
    swap = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 4, 2]))(reshape)
    
    # Flatten and fully connected layers
    flat = Flatten()(swap)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model