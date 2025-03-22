import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 Convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Branch 2: 5x5 Convolution
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    
    # Add branches
    add_layer = Add()([conv1, conv2])
    
    # Branch 1: Max Pooling for 3x3 Conv
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Branch 2: Max Pooling for 5x5 Conv
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    
    # Concatenate branches
    concat = Concatenate(axis=-1)([pool1, pool2])
    
    # Batch normalization and Flatten
    batch_norm = BatchNormalization()(concat)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    attention = Dense(2, activation='softmax')(dense1)  # Softmax for attention weights
    
    # Multiply attention weights with branch outputs
    weighted_output1 = keras.layers.multiply([dense2, attention])
    weighted_output2 = keras.layers.multiply([dense2, keras.layers.Lambda(lambda x: 1 - x)(attention)])
    
    # Final dense layer for probability distribution
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=input_layer, outputs=[output_layer, weighted_output1, weighted_output2])
    
    return model

model = dl_model()
model.summary()