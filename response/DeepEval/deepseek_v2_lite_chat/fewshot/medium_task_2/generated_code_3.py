import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Branch path
    conv2 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(inputs)
    
    # Combine paths
    add_layer = Add()([pool1, conv2])
    
    # Flatten and fully connected layers
    flatten = Flatten()(add_layer)
    fc1 = Dense(units=512, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(fc1)
    
    # Model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()