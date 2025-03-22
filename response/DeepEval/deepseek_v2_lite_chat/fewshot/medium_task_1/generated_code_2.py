import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    
    # Max-pooling layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Add main path output to branch path output
    add_layer = Add()([pool, conv2])
    
    # Flatten and pass through two fully connected layers
    flatten = Flatten()(add_layer)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model