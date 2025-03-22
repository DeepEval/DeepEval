import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.callbacks import EarlyStopping

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the images for the convolutional layers
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

# Define the model architecture
def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Encoder
    x = inputs
    for _ in range(3):
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Middle block
    x = Conv2D(64, (1, 1), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    
    # Dropout layer for regularization
    x = Dropout(0.5)(x)
    
    # Decoder
    x = Dense(512)(x)
    x = Reshape((8, 8, 32))(x)
    for _ in range(3):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2DTranspose(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
    
    # Output layer
    outputs = Conv2D(num_classes, (1, 1), padding='same')(x)
    outputs = Flatten()(outputs)
    outputs = Dense(10, activation='softmax')(outputs)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create and compile the model
model = dl_model()
model.summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', loss)
print('Test accuracy:', accuracy)