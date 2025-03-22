import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First max pooling layer
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    
    # Second max pooling layer
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    
    # Third max pooling layer
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    
    # Flatten the output from each pooling layer
    flat1 = Flatten()(maxpool1)
    flat2 = Flatten()(maxpool2)
    flat3 = Flatten()(maxpool3)
    
    # Concatenate the flattened outputs
    concatenated = Concatenate()( [flat1, flat2, flat3] )
    
    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(concatenated)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()