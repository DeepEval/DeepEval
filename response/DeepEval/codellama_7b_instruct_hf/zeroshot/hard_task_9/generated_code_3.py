from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Concatenate, Add
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator


def dl_model():
    # Load the VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the base model layers
    base_model.trainable = False

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch
    x = base_model.output
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Define the second branch
    x = base_model.output
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Define the third branch
    x = base_model.output
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Concatenate the outputs from the three branches
    x = Concatenate()([x, x, x])

    # Add a 1x1 convolution layer to adjust the output dimensions
    x = Conv2D(128, (1, 1), activation='relu')(x)

    # Define the main path
    x = base_model.output
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Add the main path and the branch together
    x = Add()([x, x])

    # Define the output layer
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('data/train', target_size=(32, 32), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory('data/test', target_size=(32, 32), batch_size=32, class_mode='categorical')

    model.fit(train_generator, epochs=10, validation_data=test_generator)

    return model