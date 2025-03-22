import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(conv2)
    
    # Concatenate the outputs of convolutional layers along the channel dimension
    concat = Concatenate()([conv3, conv2, conv1])
    
    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=1024, activation='relu')(flatten)
    dense2 = Dense(units=512, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check the summary of the model
model.summary()

# Assuming you have a training dataset 'train_data' and labels 'train_labels'
# and a testing dataset 'test_data' and labels 'test_labels'
# Train the model
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))