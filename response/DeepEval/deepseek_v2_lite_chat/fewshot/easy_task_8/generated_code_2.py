import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise Separable Convolutional Layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 1x1 Convolutional Layer for feature extraction
    conv_feature_extraction = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
    
    # Max Pooling Layer
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
    
    # Dropout Layer
    dropout1 = Dropout(0.25)(conv_feature_extraction)
    
    # Flatten Layer
    flatten = Flatten()(dropout1)
    
    # Fully Connected Layer
    dense1 = Dense(units=128, activation='relu')(flatten)
    
    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Model Construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model