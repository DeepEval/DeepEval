import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1
    def branch1():
        # Block 1
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv)
        avgpool = GlobalAveragePooling2D()(maxpool)
        
        # Fully connected layers to generate weights
        fc1 = Dense(units=512, activation='relu')(avgpool)
        fc2 = Dense(units=1024, activation='relu')(fc1)
        
        return fc2
    
    branch1_output = branch1()
    
    # Branch 2
    def branch2():
        # Block 2
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv)
        avgpool = GlobalAveragePooling2D()(maxpool)
        
        # Fully connected layers to generate weights
        fc1 = Dense(units=512, activation='relu')(avgpool)
        fc2 = Dense(units=1024, activation='relu')(fc1)
        
        return fc2
    
    branch2_output = branch2()
    
    # Concatenate and process the outputs
    concat_layer = concatenate([branch1_output, branch2_output])
    flatten = Flatten()(concat_layer)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=[input_layer] * 2, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.summary()