import keras
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First Convolutional Layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Global Average Pooling Layer
    avg_pool = GlobalAveragePooling2D()(conv1)
    
    # Fully Connected Layer 1
    dense1 = Dense(units=512, activation='relu')(avg_pool)
    
    # Fully Connected Layer 2
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model