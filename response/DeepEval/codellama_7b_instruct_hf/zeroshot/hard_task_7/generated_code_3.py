from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, DepthwiseSeparableConv2D, Flatten, Dense
from keras.applications import MNIST
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = MNIST.load_data()

# Normalize the data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert the labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create the data generators for the training and testing sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

# Define the input shape
input_shape = (28, 28, 1)

# Define the model
model = Model(
    inputs=Input(shape=input_shape),
    outputs=Block1(Block2(Lambda(lambda x: tf.split(x, 2, axis=-1)))))

# Add the initial convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

# Add the first block
model.add(Block1(Conv2D(32, (3, 3), activation='relu')))

# Add the second block
model.add(Block2(Conv2D(64, (3, 3), activation='relu')))

# Add the third block
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add the fourth block
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add the flatten layer
model.add(Flatten())

# Add the fully connected layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)