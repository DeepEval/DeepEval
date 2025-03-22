from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import Cifar10

# load CIFAR-10 dataset
cifar10 = Cifar10()

，增加dl_model 和 return model
def dl_model():
    # define input shape and number of classes
    input_shape = (32, 32, 3)
    num_classes = 10

    # define the model
    input_layer = Input(shape=input_shape)

    # main pathway
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    output_layer = Concatenate()([conv1, conv2, conv3, maxpool])

    # parallel branch
    conv4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv6 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output_branch = Concatenate()([conv4, conv5, conv6])

    # additive fusion
    output_layer = Add()([output_layer, output_branch])

    # batch normalization and flatten
    bath_norm = BatchNormalization()(output_layer)
    flatten_layer = Flatten()(bath_norm)

    # fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=num_classes, activation='softmax')(dense1)

    # define the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(cifar10.images, cifar10.labels, epochs=10, batch_size=32)

# evaluate the model
loss, accuracy = model.evaluate(cifar10.images, cifar10.labels)
print(f'Test loss: {loss:.3f}')
print(f'Test accuracy: {accuracy:.3f}')