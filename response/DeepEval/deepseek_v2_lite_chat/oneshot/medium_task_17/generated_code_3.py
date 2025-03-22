import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Permute, Reshape
from keras.models import Model

def dl_model():
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape to three groups, channels per group, then shuffle channels
    reshape_layer = Permute((2, 1))(input_layer)
    reshape_layer = Reshape((3, 32, 32, 1))(reshape_layer)
    
    # Conv2D + Pooling
    conv1 = Conv2D(32, (3, 3), activation='relu')(reshape_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Additional Block
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Flatten and Fully Connected Layers
    flat = Flatten()(pool2)
    dense1 = Dense(128, activation='relu')(flat)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Model Construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()