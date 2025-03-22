import keras
from keras.layers import Input, Conv2D, MaxPool2D, Add, Flatten, Dense, GlobalAveragePooling2D, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch 1: 3x3 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    # Branch 2: 5x5 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_layer)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    # Add branches
    add_layer = Add()([pool1, pool2])
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(add_layer)
    
    # Fully connected layers
    dropout1 = Dropout(0.5)(avg_pool)
    dense1 = Dense(units=128, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense1)
    dense2 = Dense(units=10, activation='softmax')(dropout2)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=dense2)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()