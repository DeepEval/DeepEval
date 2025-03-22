import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Concatenate, Activation, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Features extraction
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Path1: Global average pooling
    avg_pool1 = GlobalAveragePooling2D()(pool1)
    fc1_1 = Dense(units=128, activation='relu')(avg_pool1)
    
    # Path2: Global max pooling
    maxpool1 = GlobalMaxPooling2D()(pool1)
    fc1_2 = Dense(units=128, activation='relu')(maxpool1)
    
    # Add and activation for channel attention weights
    attention_weights = Add()([fc1_1, fc1_2])
    activation = Activation('sigmoid')(attention_weights)
    
    # Element-wise multiplication with original features
    attention_features = Activation('relu')(conv1) * activation
    
    # Block 2: Spatial features extraction
    avg_pool2 = GlobalAveragePooling2D()(attention_features)
    fc2 = Dense(units=64, activation='relu')(avg_pool2)
    
    maxpool2 = GlobalMaxPooling2D()(attention_features)
    fc3 = Dense(units=64, activation='relu')(maxpool2)
    
    # Concatenate and normalize
    concat = Concatenate(axis=-1)([fc2, fc3])
    normalize = Dense(units=64, activation='relu')(concat)
    normalized_features = Activation('sigmoid')(normalize)
    
    # Element-wise multiplication with channel attention features
    final_features = Activation('relu')(attention_features) * normalized_features
    
    # Additional branch for channel alignment
    conv3 = Conv2D(filters=3, kernel_size=(1, 1), activation='sigmoid')(final_features)
    output_layer = Add()([conv1, conv3])
    
    # Classification through a fully connected layer
    dense = Dense(units=1000, activation='softmax')(output_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')