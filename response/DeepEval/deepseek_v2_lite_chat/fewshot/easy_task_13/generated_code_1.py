import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layers
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Dropout for regularization
    conv1_1 = Dropout(0.2)(conv1_1)
    conv1_2 = Dropout(0.2)(conv1_2)
    
    # 3x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 1), padding='same')(conv1_1)
    
    # Dropout for regularization
    conv2 = Dropout(0.2)(conv2)
    
    # 1x3 convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(1, 3), padding='valid')(conv2)
    
    # Dropout for regularization
    conv3 = Dropout(0.2)(conv3)
    
    # 1x1 convolutional layer to adjust channels
    reshape = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(conv3)
    reshape = Flatten()(reshape)
    
    # Combine features using an addition operation
    add = Add()([conv1_2, conv2, conv3, reshape])
    
    # Flatten and fully connected layers
    flat = Flatten()(add)
    dense = Dense(units=128, activation='relu')(flat)
    
    # Output layer
    output = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()