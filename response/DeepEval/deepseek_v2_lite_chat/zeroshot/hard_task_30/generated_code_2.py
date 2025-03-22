import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, concatenate, Lambda, Dense

# Load and prepare the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Input shape
input_shape = (32, 32, 3)

# Dual-path structure model
def dual_path_block(inputs, num_filters):
    conv1 = Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same')(conv1)
    branch = Conv2D(num_filters * 2, (1, 1), activation='relu', padding='same')(inputs)
    
    addition = Add()([conv2, branch])
    return activation(addition)

def model(input_shape):
    # First block
    main_path = dual_path_block(Input(input_shape), 32)
    branch_path = Conv2D(32, (1, 1), activation='relu', padding='same')(Input(input_shape))

    combined = Add()([main_path, branch_path])

    # Second block
    split = Lambda(lambda x: tf.split(x, 3, axis=3))(combined)
    
    # Different kernel sizes for depthwise separable convolutional layers
    ds_conv1 = Conv2D(64, (1, 1), use_bias=False, padding='same')(split[0])
    ds_conv2 = Conv2D(64, (3, 3), use_bias=False, padding='same')(split[1])
    ds_conv3 = Conv2D(64, (5, 5), use_bias=False, padding='same')(split[2])
    
    # Concatenate features from different kernel sizes
    concat = concatenate([ds_conv1, ds_conv2, ds_conv3])

    # Fully connected layers
    fc1 = Dense(512, activation='relu')(concat)
    output = Dense(10, activation='softmax')(fc1)

    # Model
    model = Model(inputs=[main_path, branch_path], outputs=output)
    return model

# Create the model
model = dl_model(input_shape)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([x_train, x_train], y_train, epochs=10, validation_data=([x_test, x_test], y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate([x_test, x_test], y_test, verbose=2)
print('\nTest accuracy:', test_acc)