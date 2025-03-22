import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    
    # First branch: Local feature extraction
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    branch1 = BatchNormalization()(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(branch1)
    branch1 = BatchNormalization()(branch1)
    
    # Second branch: Downsampling using average pooling
    branch2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(branch2)
    branch2 = BatchNormalization()(branch2)
    
    # Third branch: Downsampling using average pooling
    branch3 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(branch3)
    branch3 = BatchNormalization()(branch3)
    
    # Pass through 3x3 convolutional layer and transpose convolutional layer
    branch2_expanded = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(branch2)
    branch3_expanded = keras.layers.Lambda(lambda x: keras.backend.expand_dims(x, axis=-1))(branch3)
    branch2 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(branch2_expanded)
    branch3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(branch3_expanded)
    branch2 = BatchNormalization()(branch2)
    branch3 = BatchNormalization()(branch3)
    
    # Concatenate outputs of all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Final 1x1 convolutional layer
    final_layer = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(concatenated)
    final_layer = BatchNormalization()(final_layer)
    
    # Fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(final_layer)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=dense)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Optional: Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)