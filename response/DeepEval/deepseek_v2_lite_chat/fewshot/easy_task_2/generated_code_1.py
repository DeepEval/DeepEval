import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(224, 224, 3))
    
    # Feature extraction layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)
    
    # Flatten and fully connected layers
    flatten = Flatten()(pool3)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dropout1 = Dropout(rate=0.5)(dense1)
    dense2 = Dense(units=128, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(dense2)
    
    # Output layer
    output_layer = Dense(units=1000, activation='softmax')(dropout2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()