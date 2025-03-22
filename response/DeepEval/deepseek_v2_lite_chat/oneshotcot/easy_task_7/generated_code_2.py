import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Main path
    x = inputs
    
    # Two convolution blocks to increase feature width
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Dropout to prevent overfitting
    x = Dropout(rate=0.5)(x)
    
    # Convolution layer to restore number of channels
    x = Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(x)
    
    # Branch path
    branch_x = inputs
    
    # Combine outputs from both paths
    combined = Concatenate()([x, branch_x])
    
    # Batch normalization and flattening layers
    x = BatchNormalization()(combined)
    x = Flatten()(x)
    
    # Fully connected layers
    x = Dense(units=256, activation='relu')(x)
    x = Dense(units=128, activation='relu')(x)
    outputs = Dense(units=10, activation='softmax')(x)  # 10 classes for MNIST
    
    # Construct the model
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])