from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

def block(input_tensor, filters):
    # Elevate dimension
    x = Conv2D(filters, (1, 1), activation='relu', padding='same')(input_tensor)
    
    # Depthwise separable convolution
    x = DepthwiseConv2D((3, 3), activation='relu', padding='same')(x)
    
    # Reduce dimension
    x = Conv2D(filters, (1, 1), activation='relu', padding='same')(x)
    
    # Add the input to the block output
    x = Add()([x, input_tensor])
    return x

def dl_model():
    input_shape = (28, 28, 1)
    num_classes = 10
    filters = 32

    # Define input
    inputs = Input(shape=input_shape)

    # Create three branches
    branch1 = block(inputs, filters)
    branch2 = block(inputs, filters)
    branch3 = block(inputs, filters)

    # Concatenate the branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Flatten and add a fully connected layer
    x = Flatten()(concatenated)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))