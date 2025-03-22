from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Multiply, Add, Flatten, Concatenate, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    input_shape = (32, 32, 3)
    num_classes = 10
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    # Path 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(x)
    gap_dense1 = Dense(64, activation='relu')(gap)
    gap_dense2 = Dense(32, activation='relu')(gap_dense1)
    
    # Path 2: Global Max Pooling
    gmp = GlobalMaxPooling2D()(x)
    gmp_dense1 = Dense(64, activation='relu')(gmp)
    gmp_dense2 = Dense(32, activation='relu')(gmp_dense1)
    
    # Combine channel features
    channel_features = Add()([gap_dense2, gmp_dense2])
    
    # Activation to generate channel attention weights
    channel_attention = Dense(32, activation='sigmoid')(channel_features)
    
    # Apply channel attention
    channel_attention = Multiply()([x, channel_attention])
    
    # Spatial features extraction
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attention)
    
    # Concatenate spatial features
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Combine spatial and channel features
    combined_features = Multiply()([channel_attention, spatial_features])
    
    # Flatten and fully connected layer for final output
    flattened = Flatten()(combined_features)
    outputs = Dense(num_classes, activation='softmax')(flattened)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example of compiling the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Example of training the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))