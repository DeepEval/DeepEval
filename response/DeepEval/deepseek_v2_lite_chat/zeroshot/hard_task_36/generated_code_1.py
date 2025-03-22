from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Add, GlobalAveragePooling2D, Dense
from keras.datasets import mnist
from keras.utils import to_categorical


def dl_model():

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Define the main pathway
    input_pathway = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_pathway)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    # Define the branch pathway
    input_branch = Input(shape=(14, 14, 32))  # Assuming max pooling reduces to 14x14
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_branch)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)

    # Fusion pathway
    fusion = Add()([drop1, conv2])
    fusion = Conv2D(64, (3, 3), activation='relu', padding='same')(fusion)
    fusion = MaxPooling2D(pool_size=(2, 2))(fusion)
    fusion = Dropout(0.5)(fusion)

    # Final pathway for classification
    flat1 = GlobalAveragePooling2D()(fusion)
    dense1 = Dense(128, activation='relu')(flat1)
    output = Dense(10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=[input_pathway, input_branch], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the compiled model
    return model

# Example usage:
model = dl_model()
model.fit([x_train, x_train_branch], y_train, validation_data=([x_val, x_val_branch], y_val), epochs=10)