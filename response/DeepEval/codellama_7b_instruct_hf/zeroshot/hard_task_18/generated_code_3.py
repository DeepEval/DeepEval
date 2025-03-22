from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

# Create an input layer with shape (32, 32, 3) for the CIFAR-10 images
input_layer = Input(shape=(32, 32, 3))

# Create the first sequential block
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)

# Create the second sequential block
x = Conv2D(64, (3, 3), activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

# Create the model
model = Model(inputs=input_layer, outputs=x)


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))