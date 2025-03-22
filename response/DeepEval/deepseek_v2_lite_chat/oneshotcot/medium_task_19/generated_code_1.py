import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 1x1 convolution for dimensionality reduction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch2)
    
    # Branch 3: 1x1 convolution followed by 5x5 convolution
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(branch3)
    
    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(branch4)
    
    # Concatenate the outputs of the four branches
    concatenated = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated)
    flat = Flatten()(batch_norm)
    
    # Two fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(flat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])