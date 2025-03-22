import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
    
    # Max pooling
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    
    # Paths for the block
    path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(maxpool1)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool1)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(maxpool1)
    
    # Concatenate the paths
    concatenated = Concatenate()([path1, path2, path3])
    
    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)
    
    # Flatten layer
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()