from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.optimizers import Adam

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Input shape
    input_shape = (28, 28, 1)

    # Reshape data
    x_train = x_train.astype('float32').reshape(x_train.shape[0], 28, 28, 1) / 255.0
    x_test = x_test.astype('float32').reshape(x_test.shape[0], 28, 28, 1) / 255.0

    # Convert labels to binary
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    # Main path
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Branch path
    branch_inputs = Conv2D(32, (3, 3), activation='relu')(inputs)

    # Combine paths
    x = Add()([x, branch_inputs])

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)  # Assuming 10 classes for MNIST

    # Model
    model = Model(inputs=[inputs, branch_inputs], outputs=outputs)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
model = dl_model()