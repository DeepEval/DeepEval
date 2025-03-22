import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Flatten, Concatenate
from keras.layers import Conv2DTranspose, LeakyReLU, UpSampling2D
from keras.optimizers import Adam

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model architecture
input_layer = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2D(32, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

x = Flatten()(x)

concat_layer = Concatenate()([x, input_layer])

x = Conv2D(64, (1, 1), activation='relu')(concat_layer)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2DTranspose(3, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

x = Conv2D(3, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = ReLU()(x)

output_layer = Dense(10, activation='softmax')(x)

# Compile the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy}')

# Return the trained model
return model