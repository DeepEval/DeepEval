import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Compress the input with global average pooling
    conv1 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = GlobalAveragePooling2D()(pool1)
    
    # Fully connected layer
    dense1 = Dense(512, activation='relu')(pool1)
    
    # Reshape to match the input shape for element-wise multiplication
    reshape1 = Reshape((512,))(dense1)
    
    # Element-wise multiplication with the input feature map
    multiply1 = keras.layers.multiply([reshape1, input_layer])
    
    # Fully connected layer
    dense2 = Dense(256, activation='relu')(multiply1)
    
    # Reshape to match the input shape for the final classification
    reshape2 = Reshape((256,))(dense2)
    
    # Fully connected layer
    output_layer = Dense(10, activation='softmax')(reshape2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model