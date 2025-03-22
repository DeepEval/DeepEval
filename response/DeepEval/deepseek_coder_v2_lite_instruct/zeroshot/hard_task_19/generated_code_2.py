import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the main path
    input_layer = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Define the branch path
    branch = GlobalAveragePooling2D()(x)
    branch = Dense(128, activation='relu')(branch)
    branch = Dense(64, activation='relu')(branch)
    branch_weights = Dense(128, activation='sigmoid')(branch)
    branch_weights = tf.reshape(branch_weights, (-1, 1, 1, 128))
    branch_output = Multiply()([x, branch_weights])

    # Combine the outputs of both paths
    combined = Add()([x, branch_output])

    # Add final fully connected layers for classification
    combined = GlobalAveragePooling2D()(combined)
    combined = Dense(64, activation='relu')(combined)
    output_layer = Dense(10, activation='softmax')(combined)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Example usage
model = dl_model()
model.summary()