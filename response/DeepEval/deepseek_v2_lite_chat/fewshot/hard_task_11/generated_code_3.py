import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main pathway with 1x1 convolution
    x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    
    # Branch with 1x1, 1x3, and 3x1 convolutions
    branch_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(inputs)
    branch_1x3 = Conv2D(filters=64, kernel_size=(1, 3), padding='same', strides=(1, 1))(inputs)
    branch_3x1 = Conv2D(filters=64, kernel_size=(3, 1), padding='same', strides=(1, 1))(inputs)
    
    # Concatenate the outputs from the two pathways
    concat = Concatenate(axis=-1)([x, branch_1x1, branch_1x3, branch_3x1])
    
    # Additional 1x1 convolution to match the channel size of the input
    x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concat)
    
    # Fully connected layers for classification
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    outputs = Dense(units=10, activation='softmax')(x)
    
    # Model architecture
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])