import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(32, 32),
    batch_size=32,
    class_mode='categorical'
)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add a global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a dense layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)

# Add a dense layer with 64 units and ReLU activation
x = Dense(64, activation='relu')(x)

# Add a dense layer with 10 units and softmax activation
x = Dense(10, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=x)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)


loss, accuracy = model.evaluate(test_generator)
print('Test loss:', loss)
print('Test accuracy:', accuracy)