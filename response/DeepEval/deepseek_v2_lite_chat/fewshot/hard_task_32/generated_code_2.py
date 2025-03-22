import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Concatenate, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Dropout(0.5)(branch1)
    
    # Branch 2
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    branch2 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(0.5)(branch2)
    
    # Branch 3
    branch3 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch3)
    branch3 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Dropout(0.5)(branch3)
    
    # Concatenate and process through FC layers
    concat = Concatenate(axis=-1)([branch1, branch2, branch3])
    dense1 = Dense(128, activation='relu')(concat)
    dense2 = Dense(10, activation='softmax')(dense1)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model

# Instantiate and return the model
model = dl_model()
model.summary()