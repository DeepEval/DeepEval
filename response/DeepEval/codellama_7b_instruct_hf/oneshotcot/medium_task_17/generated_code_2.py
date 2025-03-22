import keras
from keras.layers import Input, Dense, Flatten, Permute, Reshape
from keras.models import Model
from keras.applications.vgg16 import VGG16


def dl_model():
    
    # Load the VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Freeze the layers
    vgg_model.trainable = False

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the output shape
    output_shape = (10,)

    # Define the model
    model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[-1].output)
    model.summary()

    # Define the reshape layers
    reshaped_input = Reshape(target_shape=(input_shape[0], input_shape[1], input_shape[2], 1))(model.input)
    reshaped_input = Permute(dims=(0, 2, 3, 1))(reshaped_input)
    reshaped_output = Reshape(target_shape=(output_shape[0], 1))(model.output)

    # Define the flatten layer
    flattened_output = Flatten()(reshaped_output)

    # Define the fully connected layers
    dense1 = Dense(512, activation='relu')(flattened_output)
    dense2 = Dense(256, activation='relu')(dense1)
    output = Dense(output_shape[0], activation='softmax')(dense2)

    # Define the model
    model = Model(inputs=reshaped_input, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    return model