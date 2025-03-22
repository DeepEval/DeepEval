import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Conv2DTranspose, Add, concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Feature extraction part
    def extract_features(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        y = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        z = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(y)
        w = MaxPooling2D(pool_size=(2, 2), padding='same')(z)
        return [x, y, z, w]
    
    # Enhancement part
    def enhance_features(features):
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(features[0])
        y = Dropout(0.5)(x)
        z = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(y)
        w = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(features[2])
        return [x, y, z, w]
    
    # Upsampling and restoration part
    def upscale_features(enhanced_features):
        x = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(enhanced_features[0])
        x = concatenate([x, enhanced_features[1]], axis=3)
        y = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        y = concatenate([y, enhanced_features[2]], axis=3)
        z = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(y)
        z = concatenate([z, enhanced_features[3]], axis=3)
        return z
    
    # Output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), activation='softmax')(upscale_features(enhance_features([extract_features(input_layer)])))
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()