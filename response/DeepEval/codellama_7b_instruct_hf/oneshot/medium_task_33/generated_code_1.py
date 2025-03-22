import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# define the input shape
input_shape = (32, 32, 3)

# define the model
model = keras.Sequential([
    Lambda(lambda x: tf.split(x, 3, axis=3)),
    Conv2D(64, (1, 1), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (5, 5), activation='relu'),
    Concatenate(),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory('train_data', target_size=(32, 32), batch_size=32, class_mode='categorical')
model.fit(train_generator, epochs=10, validation_data=train_generator)

# evaluate the model
test_generator = datagen.flow_from_directory('test_data', target_size=(32, 32), batch_size=32, class_mode='categorical')
loss, accuracy = model.evaluate(test_generator)
print('Loss: {}, Accuracy: {}'.format(loss, accuracy))