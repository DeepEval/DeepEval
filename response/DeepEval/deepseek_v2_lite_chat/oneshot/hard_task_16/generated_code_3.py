import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups for three parallel paths
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    split2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    split3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(inputs)
    
    # Convolutional layers
    conv1 = Conv2D(32, (1, 1), activation='relu')(split1[0])
    conv2 = Conv2D(32, (3, 3), activation='relu')(split2[1])
    conv3 = Conv2D(32, (1, 1), activation='relu')(split3[2])
    
    # Concatenate the outputs from the three paths
    concat = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Transition Convolution
    transition = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(concat)
    
    # Batch normalization and Max Pooling
    batch_norm = BatchNormalization()(transition)
    max_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(batch_norm)
    
    # Fully connected layers
    flat = Flatten()(max_pool)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    
    # Channel matching and reshaping
    weights = Dense(max_pool.shape[3]*max_pool.shape[1]*max_pool.shape[2])(dense2)
    reshaped_weights = keras.backend.reshape(weights, (-1, max_pool.shape[3], max_pool.shape[1], max_pool.shape[2]))
    output_weights = keras.layers.multiply([reshaped_weights, max_pool])
    
    # Addition of main path and branch outputs
    main_output = Dense(10, activation='softmax')(dense1)
    branch_output = Lambda(lambda x: keras.backend.sum(x, axis=1), output_shape=(1,))(output_weights)
    total_output = keras.backend.sum([main_output, branch_output], axis=0)
    
    # Model construction
    model = Model(inputs=inputs, outputs=total_output)
    
    return model