import keras
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # 1x1 convolutional layer to compress channels
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Parallel convolutional paths
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Concatenate the outputs of the parallel paths
    concat_layer = Concatenate()([conv2, conv3])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concat_layer)
    flat_layer = Flatten()(batch_norm)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flat_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Construct the model
model = dl_model()
model.summary()