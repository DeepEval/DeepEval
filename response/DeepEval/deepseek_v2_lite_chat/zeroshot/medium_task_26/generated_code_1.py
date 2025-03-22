import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Compression layer
    x = Conv2D(16, (1, 1), activation='relu')(input_layer)
    
    # Expansion layers
    x = Conv2D(32, (1, 1), activation='relu')(x)  # 1x1 conv
    x = Conv2D(32, (3, 3), activation='relu')(x)  # 3x3 conv
    
    # Concatenation
    x = concatenate([x, input_layer])
    
    # Flattening
    x = Flatten()(x)
    
    # Classification layers
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)  # Assuming 10 classes
    
    # Model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])