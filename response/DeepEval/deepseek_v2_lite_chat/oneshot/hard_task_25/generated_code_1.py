import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, ZeroPadding2D, Conv2DTranspose

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the data to fit the model's expected input shape
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

# Function to create the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch path
    def branch(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool)
        upconv = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='valid')(conv2)
        return Concatenate()([input_tensor, conv1, upconv, conv2])
    
    branch_output = branch(conv1)
    branch_output = BatchNormalization()(branch_output)
    branch_output = Flatten()(branch_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(branch_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)