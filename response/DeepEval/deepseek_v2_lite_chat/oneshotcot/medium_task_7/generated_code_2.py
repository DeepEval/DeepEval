import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    
    # Third convolutional layer with different kernel size
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(conv2)
    
    # Separate convolutional layer processing the input directly
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Concatenate the outputs from all paths
    concat = Concatenate()([conv3, conv4])
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2))(concat)
    
    # Batch normalization, flattening, and dense layers
    batch_norm = BatchNormalization()(pool1)
    flatten = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()