import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, Add, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the input shape
input_shape = (32, 32, 3)

# Function to create the model
def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # First parallel path
    path1 = GlobalAveragePooling2D()(x)
    path1 = Dense(512, activation='relu')(path1)
    path1 = Dense(10, activation='softmax')(path1)
    
    # Second parallel path
    path2 = GlobalMaxPooling2D()(x)
    path2 = Dense(512, activation='relu')(path2)
    path2 = Dense(10, activation='softmax')(path2)
    
    # Add the outputs of both paths
    outputs = Add()([path1, path2])
    
    # Activation function to generate channel attention weights
    attention = Activation('sigmoid')(Dense(1)(outputs))
    
    # Apply attention weights to the original features
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Average and max pooling operations
    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    
    # Concatenate along the channel dimension
    concat = Concatenate()([avg_pool, max_pool])
    
    # Fully connected layer for the final output
    output = Dense(10, activation='softmax')(concat)
    
    # Compile the model
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model

# Construct the model
model = dl_model()

# Print model summary
model.summary()