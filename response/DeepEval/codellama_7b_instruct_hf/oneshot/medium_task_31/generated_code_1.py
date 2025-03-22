import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# Load the VGG16 model
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the convolutional layers
    conv1_1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the pooling layers
    pool1_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1_1)
    pool1_2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(conv1_2)
    pool1_3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(conv1_3)

    # Define the concatenation layer
    concat_layer = Concatenate()([pool1_1, pool1_2, pool1_3])

    # Define the batch normalization layer
    batch_norm = BatchNormalization()(concat_layer)

    # Define the flatten layer
    flatten_layer = Flatten()(batch_norm)

    # Define the fully connected layers
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(64, activation='relu')(dense1)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model


# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess the images
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Generate the data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
validation_generator = test_datagen.flow(X_val, y_val, batch_size=32)

# Train the model
model.fit(train_generator, epochs=10, validation_data=(validation_generator))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)