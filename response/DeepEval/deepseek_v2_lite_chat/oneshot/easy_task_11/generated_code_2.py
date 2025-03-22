import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.layers import Dropout

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Convolutional layer with 5x5 window and 3x3 stride for dimensionality reduction
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid')(input_layer)
    
    # 1x1 convolutional layer to enhance depth and introduce nonlinearity
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    
    # Flatten the feature maps
    flatten_layer = Flatten()(conv)
    
    # Dropout layer to mitigate overfitting between fully connected layers
    dropout_layer = Dropout(0.5)(flatten_layer)
    
    # Fully connected layers for additional feature processing
    dense1 = Dense(units=128, activation='relu')(dropout_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Output layer with softmax activation for multi-class classification
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])