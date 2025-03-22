import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, UpSampling2D, Conv2DTranspose
from keras.layers.merge import Add

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the input shape
input_shape = (32, 32, 3)

# Define the model
def dl_model():
    # Input layer
    input_layer = Input(shape=input_shape)

    # Convolutional and max-pooling layers
    def extract_features(input_tensor):
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        return conv3, pool3

    # Processing block with Dropout
    def enhance_features(input_tensor):
        drop1 = Dropout(0.5)(input_tensor)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop1)
        drop2 = Dropout(0.5)(conv4)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
        return conv5

    # Upsampling block
    def restore_spatial_info(input_tensor, conv3):
        up1 = UpSampling2D(size=(2, 2))(input_tensor)
        merge1 = Add()([conv5, up1])
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
        merge2 = Add()([conv6, conv3])
        conv7 = Conv2D(64, (3, 3), activation='relu')(merge2)
        conv8 = Conv2D(3, (1, 1), activation='sigmoid')(conv7)
        return conv8

    # Concatenate the features
    conv3, pool3 = extract_features(input_layer)
    conv5 = enhance_features(pool3)
    model_output = restore_spatial_info(conv3, conv5)

    # Output layer
    output_layer = Conv2D(units=10, kernel_size=(1, 1), activation='softmax')(model_output)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)