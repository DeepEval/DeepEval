import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, ZeroPadding2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # First branch: Two sequential 3x3 convolutional layers for local feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch1)
    
    # Second branch: Average pooling, 3x3 convolutional layer, transposed convolutional layer
    branch2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(branch2)
    
    # Third branch: Average pooling, 3x3 convolutional layer, transposed convolutional layer
    branch3 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(branch3)
    
    # Concatenate the outputs of the three branches
    concat = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolutional layer to refine the concatenated output
    conv_out = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    
    # Fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(conv_out)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()