from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.applications.vgg16 import VGG16



def dl_model():
    
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the four parallel convolutional paths
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_shape)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1_2)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_1)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2_2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_1)

    # Define the max pooling path
    max_pooling = MaxPooling2D((2, 2))(conv3_2)

    # Define the concatenation path
    concat = Concatenate(axis=-1)([conv1_2, conv2_1, conv3_1, max_pooling])

    # Define the flatten path
    flatten = Flatten()(concat)

    # Define the dense path
    dense = Dense(128, activation='relu')(flatten)

    # Define the output layer
    output = Dense(10, activation='softmax')(dense)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import EarlyStopping

    # Generate data
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('path/to/train/data', target_size=(32, 32), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory('path/to/test/data', target_size=(32, 32), batch_size=32, class_mode='categorical')

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit_generator(train_generator, epochs=10, validation_data=test_generator, callbacks=[early_stopping])


    # Evaluate the model
    loss, accuracy = model.evaluate(test_generator)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    return model