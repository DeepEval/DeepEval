import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Concatenate, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer with batch normalization and ReLU activation
    conv1 = Conv2D(32, (3, 3), padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    # Global average pooling
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Fully connected layer to match the dimension of initial features
    flat1 = Flatten()(pool1)
    
    # Fully connected layer to adjust dimensions
    dense1 = Dense(units=512, activation='relu')(flat1)
    dense2 = Dense(units=256, activation='relu')(dense1)
    
    # Concatenate with initial features to generate weighted feature maps
    concat_layer = Concatenate()([dense2, conv1])
    
    # 1x1 convolution and average pooling for dimensionality reduction
    conv2 = Conv2D(64, (1, 1), padding='same')(concat_layer)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(pool2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])