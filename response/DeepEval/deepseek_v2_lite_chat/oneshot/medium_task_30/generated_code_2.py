import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Dense, BatchNormalization

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolution and max pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten and concatenate the outputs
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)
    
    concatenated = Concatenate()([flat1, flat2, flat3])
    
    # Fully connected layers
    dense1 = Dense(units=1024, activation='relu')(concatenated)
    dense2 = Dense(units=512, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and print the model
model = dl_model()
model.summary()