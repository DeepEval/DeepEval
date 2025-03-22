from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.applications import VGG16
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    def dl_model():
        # Input shape
        input_shape = (224, 224, 3)

        # Base model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        # Sequential feature extraction layers
        model = Sequential()
        model.add(base_model)
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        # Flatten feature maps
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(1000, activation='softmax'))

        return model