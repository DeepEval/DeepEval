import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Softmax

def dl_model():
    input_layer = Input(shape=(224, 224, 3))
    
    # First feature extraction layer: Convolution followed by Average Pooling
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Second feature extraction layer: Convolution followed by Average Pooling
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Three additional convolutional layers
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    
    # Additional Average Pooling layer to reduce dimensionality
    x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    
    # Flatten the feature maps
    x = Flatten()(x)
    
    # First fully connected layer with dropout
    x = Dense(units=1024, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    
    # Second fully connected layer with dropout
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    
    # Output layer with softmax activation for classification into 1,000 categories
    output_layer = Dense(units=1000, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model