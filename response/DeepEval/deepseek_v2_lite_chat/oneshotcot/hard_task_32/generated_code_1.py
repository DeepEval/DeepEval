import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the model architecture for each branch
    def block(input_tensor):
        # Depthwise separable convolution
        conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', depth_wise=True)(input_tensor)
        # 1x1 convolutional layer for feature extraction
        conv = Conv2D(filters=64, kernel_size=(1, 1))(conv)
        # Dropout layer for regularization
        conv = Dropout(0.5)(conv)
        # MaxPooling2D layer
        max_pooling = MaxPooling2D(pool_size=(2, 2))(conv)
        return max_pooling
    
    # Define the model for each branch
    def branch(input_tensor):
        branch_output = block(input_tensor)
        branch_output = BatchNormalization()(branch_output)
        branch_output = Flatten()(branch_output)
        dense1 = Dense(units=128, activation='relu')(branch_output)
        dense2 = Dense(units=64, activation='relu')(dense1)
        return dense2
    
    # Construct the three branches
    branch1 = branch(input_tensor=input_layer)
    branch2 = branch(input_tensor=input_layer)
    branch3 = branch(input_tensor=input_layer)

    # Concatenate the outputs of the three branches
    concat_layer = Concatenate()([branch1, branch2, branch3])
    
    # Process through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(concat_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)