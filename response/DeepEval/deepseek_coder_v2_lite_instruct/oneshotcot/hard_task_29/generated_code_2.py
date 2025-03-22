import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First Block
    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    
    # Branch Path
    branch = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Addition of Main and Branch Paths
    addition = Add()([conv2, branch])
    
    # Second Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1)(addition)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(addition)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4)(addition)
    
    # Flattening and Concatenation
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concatenated = keras.layers.Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model Construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model