import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, UpSampling2D, ZeroPadding2D, BatchNormalization
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Stage 1: Feature Extraction
    # Conv2D -> MaxPooling2D
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Dropout for regularization
    drop1 = Dropout(0.2)(pool2)
    
    # Stage 2: Restore Spatial Information
    # UpSampling2D -> Concatenate
    up1 = UpSampling2D(size=(2, 2))(drop1)
    concat = Concatenate()([up1, conv2])
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(concat)
    conv4 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv3)
    
    # Skip connection
    skip = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(drop1)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(concat)
    x = BatchNormalization()(x)
    x = keras.layers.add([x, skip])
    
    # Final convolution
    out = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(x)
    
    # Model
    model = Model(inputs=input_layer, outputs=out)
    
    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()