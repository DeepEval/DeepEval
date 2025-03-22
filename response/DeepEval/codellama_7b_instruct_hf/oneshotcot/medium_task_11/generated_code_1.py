import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply
from keras.models import Model
from keras.applications.vgg16 import VGG16


def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocess the images by resizing them to 32x32 pixels and normalizing the pixel values
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

    # Define the CNN architecture
    input_layer = Input(shape=(32, 32, 3))
    conv_layer = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Define the channel attention path
    channel_attention_path = GlobalAveragePooling2D()(conv_layer)
    channel_attention_path = Dense(16, activation='relu')(channel_attention_path)
    channel_attention_path = Dense(32, activation='relu')(channel_attention_path)
    channel_attention_path = Dense(1, activation='sigmoid')(channel_attention_path)
    channel_attention_path = Multiply()([conv_layer, channel_attention_path])

    # Define the spatial attention path
    spatial_attention_path = GlobalMaxPooling2D()(conv_layer)
    spatial_attention_path = Dense(16, activation='relu')(spatial_attention_path)
    spatial_attention_path = Dense(32, activation='relu')(spatial_attention_path)
    spatial_attention_path = Dense(1, activation='sigmoid')(spatial_attention_path)
    spatial_attention_path = Multiply()([conv_layer, spatial_attention_path])

    # Combine the channel and spatial attention paths
    fused_features = Multiply()([channel_attention_path, spatial_attention_path])

    # Flatten the fused features
    flatten_layer = Flatten()(fused_features)

    # Define the fully connected layers
    fc1 = Dense(128, activation='relu')(flatten_layer)
    fc2 = Dense(64, activation='relu')(fc1)
    fc3 = Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc3)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

    # Use the trained model to make predictions on new images
    predictions = model.predict(new_images)

    return model