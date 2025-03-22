import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, Add
from keras.layers import BatchNormalization, Activation

def dl_model():
    # Input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels
    
    # Input layer
    input_image = Input(shape=input_shape)
    
    # First path for feature extraction
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_image)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Second path for feature enhancement
    y = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    y = Dropout(rate=0.5)(y)
    
    y = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(y)
    y = Dropout(rate=0.5)(y)
    
    # Third path for upsampling and restoration
    z = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(y)
    z = concatenate([z, x], axis=3)
    
    z = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(z)
    z = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(z)
    z = concatenate([z, x], axis=3)
    
    z = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(z)
    z = Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=(2, 2), padding='same')(z)
    
    # Output layer
    output_layer = Conv2D(filters=10, kernel_size=(1, 1), padding='same', activation='softmax')(z)
    
    # Model
    model = Model(inputs=[input_image], outputs=[output_layer])
    
    return model

# Optional: compile and print summary of the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()