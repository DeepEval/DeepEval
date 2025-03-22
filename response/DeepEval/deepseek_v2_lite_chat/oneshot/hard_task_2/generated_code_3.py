import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    split2 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(split1[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(split1[1])
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(split1[2])
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(split2[0])
    conv5 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(split2[1])
    conv6 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(split2[2])
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    
    # Concatenate the outputs from the three paths
    main_path = Concatenate(axis=-1)([pool1, pool2, pool3, pool4, pool5, pool6])
    
    # Add the original input layer to the main path
    combined = Concatenate()([input_layer, main_path])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(combined)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model