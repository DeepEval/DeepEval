import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: Convolution with 3x3 kernel
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 2: Convolution with 5x5 kernel
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine outputs of both branches
    add_layer = Concatenate()([conv1, conv2])
    
    # Global average pooling
    avg_pool = AveragePooling2D(pool_size=(2, 2))(add_layer)
    
    # Two fully connected layers
    flatten = Flatten()(avg_pool)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Softmax attention weights
    attention_weights = Dense(units=2, activation='softmax', name='attention_weights')(dense1)
    
    # Multiply outputs of branches by their corresponding weights
    weighted_output = keras.layers.multiply([dense2, attention_weights])
    
    # Final dense layer for probability distribution
    output_layer = Dense(units=10, activation='softmax')(weighted_output)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()