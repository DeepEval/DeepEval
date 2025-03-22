import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch path
    branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine outputs from main and branch paths
    combined = Add()[(1, 1)]([conv2, branch])
    
    # Block 1: Addition and pooling
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(combined)
    
    # Block 2: Three max pooling layers with varying scales
    pool2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(combined)
    pool3 = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(combined)
    pool4 = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='valid')(combined)
    
    # Flatten and concatenate
    flatten = Flatten()(Concatenate()([pool1, pool2, pool3, pool4]))
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()