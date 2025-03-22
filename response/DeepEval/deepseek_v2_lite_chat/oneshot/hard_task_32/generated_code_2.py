import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Concatenate, Dropout, BatchNormalization, Flatten, Dense

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first branch block
    def branch1():
        def block(input_tensor):
            conv = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            depthwise = Conv2D(32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last')(input_tensor)
            conv1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)
            dropout1 = Dropout(0.5)(conv1)
            return dropout1
        return block(input_tensor=input_layer)

    # Define the second branch block
    def branch2():
        def block(input_tensor):
            conv = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            depthwise = Conv2D(32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last')(input_tensor)
            conv1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)
            dropout1 = Dropout(0.5)(conv1)
            return dropout1
        return block(input_tensor=input_layer)

    # Define the third branch block
    def branch3():
        def block(input_tensor):
            conv = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
            depthwise = Conv2D(32, (3, 3), strides=(1, 1), padding='same', data_format='channels_last')(input_tensor)
            conv1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise)
            dropout1 = Dropout(0.5)(conv1)
            return dropout1
        return block(input_tensor=input_layer)

    # Compute the outputs for each branch
    branch1_output = branch1()
    branch2_output = branch2()
    branch3_output = branch3()

    # Concatenate the outputs from all branches
    concat_layer = Concatenate()(outputs=[branch1_output, branch2_output, branch3_output])

    # Process through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()