from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    # Input layer
    input_img = Input(shape=(224, 224, 3))

    # Sequential feature extraction layers
    x = Conv2D(filters=64, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu')(input_img)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    x = Conv2D(filters=192, kernel_size=(11, 11), strides=(1, 1), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # Additional convolutional layers
    x = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.4)(x)

    # Output layer
    output = Dense(1000, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_img, outputs=output)

    return model

# Print the model summary
model = dl_model()
model.summary()