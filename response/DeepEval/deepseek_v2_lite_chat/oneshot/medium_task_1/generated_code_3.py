import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Input layer
input_layer = Input(shape=(32, 32, 3))

# Convolutional layers and max pooling
conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

# Concatenate the outputs of parallel paths
def block(input_tensor):
    path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
    path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')(input_tensor)
    path4 = MaxPooling2D(pool_size=(1, 1), padding='same')(input_tensor)
    output_tensor = Concatenate()([path1, path2, path3, path4])
    return output_tensor

block_output = block(pool2)
bn = BatchNormalization()(block_output)
flatten = Flatten()(bn)

# Fully connected layers
dense1 = Dense(units=128, activation='relu')(flatten)
dense2 = Dense(units=64, activation='relu')(dense1)
output_layer = Dense(units=10, activation='softmax')(dense2)

# Construct the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model

# Build the model
model = dl_model()

# Print the model summary
model.summary()

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)