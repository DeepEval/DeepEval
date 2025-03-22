import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Feature extraction through separable convolutional layers of varying sizes
    x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(x[0])
    x2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x[1])
    x3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(x[2])

    # Concatenate the outputs from three groups
    x = Concatenate()([x1, x2, x3])

    # Batch normalization
    x = BatchNormalization()(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with the CIFAR-10 dataset
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory('data/train', target_size=(32, 32), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory('data/test', target_size=(32, 32), batch_size=32, class_mode='categorical')
    model.fit_generator(train_generator, epochs=10, validation_data=test_generator)

    return model