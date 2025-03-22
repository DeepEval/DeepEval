import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, UpSampling2D, ZeroPadding2D, BatchNormalization

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Stage 1: Downsampling using convolutional and max pooling layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    
    # Stage 2: Feature extraction with more convolutional layers
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    
    # Flatten and feed into dense layers
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # Dropout to prevent overfitting
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())