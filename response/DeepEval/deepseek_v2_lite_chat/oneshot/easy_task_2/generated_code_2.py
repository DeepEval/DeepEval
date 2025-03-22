import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, BatchNormalization
from keras.layers import Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # Feature extraction layers
    conv1 = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(conv2)
    
    # Concatenate four parallel paths
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(input_tensor)
        return Concatenate()([path1, path2, path3, path4])
    
    block_output = block(pool2)
    batch_norm = BatchNormalization()(block_output)
    flatten = Flatten()(batch_norm)
    
    # Process through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Add dropout for regularization
    dropout1 = Dropout(rate=0.5)(dense2)
    
    # Output layer
    output_layer = Dense(units=1000, activation='softmax')(dropout1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()