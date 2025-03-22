import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = LeakyReLU(alpha=0.2)(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)
    
    # Branch 2
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = LeakyReLU(alpha=0.2)(branch2)
    
    # Branch 3
    branch3 = Conv2D(filters=96, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)
    branch3 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = LeakyReLU(alpha=0.2)(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch3)
    
    # Concatenate the outputs from all branches
    concat = Concatenate()( [branch1, branch2, branch3] )
    
    # Process through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model