import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Split the input into three groups along the last dimension
    split_size = x_train.shape[0] // 3
    input_layer = Lambda(lambda x: keras.backend.resize_images(x, (32, 32)))(Input(shape=(32, 32, 3)))
    groups = [
        keras.layers.Lambda(lambda x: x[:, :split_size, :, :]),
        keras.layers.Lambda(lambda x: x[:, split_size:2*split_size, :, :]),
        keras.layers.Lambda(lambda x: x[:, 2*split_size:, :, :])
    ]

    # Feature extraction layers
    for i, group in enumerate(groups):
        for j in range(3):
            if i == 0:
                conv_kernel_size = (1, 1)
            elif i == 1:
                conv_kernel_size = (3, 3)
            else:
                conv_kernel_size = (5, 5)

            group.append(Conv2D(filters=32, kernel_size=conv_kernel_size, strides=(1, 1), padding='same', activation='relu')(group[j]))
            group.append(BatchNormalization(axis=3))(group[j])
            group.append(MaxPooling2D(pool_size=(2, 2))(group[j]))

    # Concatenate and fuse the outputs
    output_tensor = Concatenate()(groups[0] + groups[1] + groups[2])
    output_tensor = BatchNormalization(axis=3)(output_tensor)
    output_tensor = Flatten()(output_tensor)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(output_tensor)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)