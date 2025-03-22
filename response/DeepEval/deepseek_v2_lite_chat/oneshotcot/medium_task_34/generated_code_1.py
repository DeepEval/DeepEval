import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Activation, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)  # Adjust based on your specific CIFAR-10 data shape
    
    # Part 1: Feature Extraction
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    
    # Part 2: Enhance Generalization
    x = Dropout(0.5)(x)  # Dropout to prevent overfitting
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    
    # Part 3: Upsampling and Restoring Spatial Information
    x = UpSampling2D(size=2)(x)  # Upsample to original size
    x = concatenate([x, x])  # Skip connections from previous layers
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = UpSampling2D(size=2)(x)  # Upsample again
    x = concatenate([x, x])  # Skip connections
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = UpSampling2D(size=2)(x)  # Upsample again
    x = concatenate([x, x])  # Skip connections
    x = Conv2D(3, (1, 1), activation='sigmoid')(x)  # 1x1 conv for probability output
    
    # Output layer
    model = Model(inputs=input_layer, outputs=x)
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate the model
model = dl_model()