import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, BatchNormalization

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the RGB codes to the range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Define the model architecture
    def create_model_path1():
        input_layer = Input(shape=(32, 32, 3))
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        return conv1, pool1
    
    def create_model_path2():
        input_layer = Input(shape=(32, 32, 3))
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
        return conv2
    
    conv1, pool1 = create_model_path1()
    conv2 = create_model_path2()
    
    # Add paths together
    add_layer = Add()([conv1, conv2])
    
    # Additional blocks to increase depth
    def block(input_tensor):
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        bn = BatchNormalization()(conv3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(bn)
        return pool2
    
    bn = BatchNormalization()(add_layer)
    pool2 = block(bn)
    flat = Flatten()(pool2)
    
    # Fully connected layers
    dense1 = Dense(units=512, activation='relu')(flat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=[conv1, conv2], outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()