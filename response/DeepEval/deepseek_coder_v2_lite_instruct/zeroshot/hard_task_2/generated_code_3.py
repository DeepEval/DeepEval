import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape and normalize the input data
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Process each group through a series of convolution layers
    processed_groups = []
    for group in split_layer:
        x = Conv2D(32, (1, 1), activation='relu')(group)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(32, (1, 1), activation='relu')(x)
        processed_groups.append(x)
    
    # Combine the processed groups using an addition operation
    combined = Add()(processed_groups)
    
    # Add the combined features back to the original input
    output = Add()([combined, inputs])
    
    # Flatten the output and pass it through a fully connected layer
    x = Flatten()(output)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=predictions)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))