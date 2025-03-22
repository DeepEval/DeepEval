from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
    
    # Max-pooling layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Ensure input and pool layer have the same dimensions for addition
    # Use a 1x1 convolution to adjust dimensions if necessary, but here it's a direct addition assuming dimensions match.
    # Add input layer to the pooled features
    added_features = Add()([input_layer, pool])
    
    # Flatten the added features
    flattened = Flatten()(added_features)
    
    # First fully connected layer
    fc1 = Dense(units=128, activation='relu')(flattened)
    
    # Second fully connected layer
    fc2 = Dense(units=64, activation='relu')(fc1)
    
    # Output layer with 10 classes
    output_layer = Dense(units=10, activation='softmax')(fc2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# To use the model, you can load the dataset and train the model as follows:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = dl_model()
model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))