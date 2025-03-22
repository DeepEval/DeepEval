import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolution
    separable_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', 
                            kernel_initializer='he_normal', activation='relu')(input_layer)
    
    # 1x1 convolutional layer for feature extraction
    feature_extraction = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', 
                                 activation='relu')(separable_conv)
    
    # Dropout layer for regularization
    dropout1 = Dropout(rate=0.5)(feature_extraction)
    dropout2 = Dropout(rate=0.5)(dropout1)
    
    # Batch normalization and max pooling
    batch_norm = BatchNormalization()(dropout2)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(batch_norm)
    
    # Flatten layer
    flatten = Flatten()(pool1)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])