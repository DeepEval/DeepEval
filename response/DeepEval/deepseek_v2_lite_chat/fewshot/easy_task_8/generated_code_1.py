import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Depthwise separable convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_uniform')(input_layer)
    
    # 1x1 convolutional layer for feature extraction
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Dropout layer after each convolutional layer to mitigate overfitting
    conv1_dropout = Dropout(0.5)(conv1)
    conv2_dropout = Dropout(0.5)(conv2)
    
    # Max pooling layer
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_dropout)
    
    # Flatten layer
    flatten = Flatten()(pool1)
    
    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    
    # Dropout layer after the fully connected layer
    dense_dropout = Dropout(0.5)(dense)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_dropout)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])