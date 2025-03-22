import keras
from keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, Multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: Average pooling
    branch1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch1_flatten = Flatten()(branch1)
    fc1_1 = Dense(units=1024, activation='relu')(branch1_flatten)
    
    # Branch 2: Global average pooling
    branch2 = GlobalAveragePooling2D()(input_layer)
    branch2_flatten = Flatten()(branch2)
    fc1_2 = Dense(units=1024, activation='relu')(branch2_flatten)
    
    # Combine branches
    concat = Concatenate()([fc1_1, fc1_2])
    fc2 = Dense(units=1024, activation='relu')(concat)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()