import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Compress input features with global average pooling
    pooled_layer = GlobalAveragePooling2D()(input_layer)
    
    # Two fully connected layers to generate weights
    dense1 = Dense(units=256, activation='relu')(pooled_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Reshape weights to align with input shape
    reshape_layer = Reshape((-1, dense2.shape[1]))(dense2)
    
    # Multiply element-wise with the input feature map
    multiplied_layer = keras.layers.multiply([reshape_layer, input_layer])
    
    # Flatten the result and pass through another fully connected layer
    flatting_layer = Flatten()(multiplied_layer)
    output_layer = Dense(units=10, activation='softmax')(flatting_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()