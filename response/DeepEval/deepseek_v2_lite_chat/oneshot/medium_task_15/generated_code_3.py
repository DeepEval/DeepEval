import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batchnorm1 = BatchNormalization()(conv1)
    activation1 = Activation('relu')(batchnorm1)
    
    # Global average pooling
    avg_pooling = MaxPooling2D(pool_size=(2, 2))(activation1)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(avg_pooling)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Reshape to match the size of initial features
    reshape = Flatten()(dense2)
    
    # Multiply with initial features to generate weighted feature maps
    concat_weighted = Concatenate()([reshape, activation1])
    
    # 1x1 convolution and average pooling
    conv_pooling = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(concat_weighted)
    avg_pooling_output = MaxPooling2D(pool_size=(4, 4))(conv_pooling)
    
    # Output layer
    output = Dense(units=10, activation='softmax')(avg_pooling_output)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])