import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Three feature extraction branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)  # 1x1 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)  # 1x1 + 3x3 convolution
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)  # 1x1 + 2*3x3 convolution
    
    # Concatenate outputs from all branches
    concatenated = Concatenate()(inputs=[branch1, branch2, branch3])
    
    # Adjust dimensions to match input image's channel size
    adjusted = Conv2D(filters=3, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Main path directly connects to the input
    main_path_input = Input(shape=input_layer.shape)
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(main_path_input)
    
    # Fusion through addition
    model = Model(inputs=main_path_input, outputs=main_path_output + adjusted)
    
    # Classification head
    flatten = Flatten()(model.layers[-1].output)  # Flatten the output tensor
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Return the constructed model
    return Model(inputs=input_layer, outputs=output_layer)

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)