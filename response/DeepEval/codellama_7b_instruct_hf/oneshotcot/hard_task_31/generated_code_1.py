import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Split
from keras.models import Model
from keras.applications.vgg16 import VGG16

# Load the VGG-16 model
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the layers of the VGG-16 model
for layer in vgg.layers:
    layer.trainable = False


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first block of the model
    main_path = Conv2D(64, (3, 3), activation='relu')(input_layer)
    branch_path = Conv2D(64, (3, 3), activation='relu')(input_layer)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)
    branch_path = MaxPooling2D(pool_size=(2, 2))(branch_path)
    output_tensor = Concatenate()([main_path, branch_path])

    # Define the second block of the model
    output_tensor = Split(axis=-1, num_split=3)(output_tensor)
    output_tensor = Concatenate()(output_tensor)
    output_tensor = Conv2D(64, (1, 1), activation='relu')(output_tensor)
    output_tensor = Conv2D(64, (3, 3), activation='relu')(output_tensor)
    output_tensor = Conv2D(64, (5, 5), activation='relu')(output_tensor)
    output_tensor = MaxPooling2D(pool_size=(2, 2))(output_tensor)
    output_tensor = Concatenate()([output_tensor, vgg.layers[-2].output])

    # Define the third block of the model
    output_tensor = Conv2D(64, (1, 1), activation='relu')(output_tensor)
    output_tensor = Conv2D(64, (3, 3), activation='relu')(output_tensor)
    output_tensor = Conv2D(64, (5, 5), activation='relu')(output_tensor)
    output_tensor = MaxPooling2D(pool_size=(2, 2))(output_tensor)
    output_tensor = Concatenate()([output_tensor, vgg.layers[-1].output])

    # Define the final output layer
    output_tensor = Flatten()(output_tensor)
    output_tensor = Dense(128, activation='relu')(output_tensor)
    output_tensor = Dense(10, activation='softmax')(output_tensor)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_tensor)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model