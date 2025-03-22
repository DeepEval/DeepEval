import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense

def dl_model():
    # Number of classes
    num_classes = 10
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        # Block 1
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Block 2
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Branch path
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        maxpool3 = MaxPooling2D(pool_size=(1, 1))(conv3)
        
        # Concatenate the outputs from both paths
        concatenated = Concatenate()(inputs=[maxpool2, maxpool3])
        
        # Batch normalization and flattening
        bn_concat = BatchNormalization()(concatenated)
        flat = Flatten()(bn_concat)
        
        # Fully connected layers
        dense1 = Dense(units=128, activation='relu')(flat)
        output = Dense(units=num_classes, activation='softmax')(dense1)
        
        # Create the model
        model = Model(inputs=input_layer, outputs=output)
        return model
    
    # Return the constructed model
    return main_path(input_layer)

# Load and prepare the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build and train the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))