import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Conv2DTranspose, Add
from keras.layers import BatchNormalization, Activation

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Feature extraction part
    x = inputs
    for _ in range(3):
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add dropout for regularization
    x = Dropout(0.5)(x)
    
    # Processing part
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    
    # Skip connections to the first part
    x = Add()([x, inputs])
    
    # Up-sampling part
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = Add()([x, inputs])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = Add()([x, inputs])
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Conv2DTranspose(3, (3, 3), strides=2, padding='same')(x)
    
    # Output layer
    outputs = Conv2D(10, (1, 1), activation='softmax')(x)
    
    # Model
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])