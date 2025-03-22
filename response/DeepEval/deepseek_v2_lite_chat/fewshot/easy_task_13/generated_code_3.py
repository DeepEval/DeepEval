import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # 1x1 convolutional layer
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    # 3x1 convolutional layer
    conv3_1 = Conv2D(filters=32, kernel_size=(3, 1), padding='same')(input_layer)
    # 1x3 convolutional layer
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 3), padding='valid')(input_layer)
    
    # 1x1 convolutional layer to match channels
    conv_expand = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Dropout for regularization
    conv1_1 = Dropout(0.5)(conv1_1)
    conv3_1 = Dropout(0.5)(conv3_1)
    conv1_3 = Dropout(0.5)(conv1_3)
    conv_expand = Dropout(0.5)(conv_expand)
    
    # Apply activation functions
    conv1_1 = keras.activations.relu(conv1_1)
    conv3_1 = keras.activations.relu(conv3_1)
    conv1_3 = keras.activations.relu(conv1_3)
    conv_expand = keras.activations.relu(conv_expand)
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    pool3 = MaxPooling2D(pool_size=(3, 3))(conv3_1)
    pool1_3 = MaxPooling2D(pool_size=(1, 3))(conv1_3)
    
    # Add all layers together
    merged = Add()([pool1, pool3, pool1_3])
    
    # Flatten and fully connected layers
    flat = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flat)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()