import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Number of classes
num_classes = 10

# Input layer
input_layer = Input(shape=(32, 32, 3))

# Main path
def main_path(input_tensor):
    conv = Conv2D(32, (3, 3), activation='relu')(input_tensor)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    avg_pool = GlobalAveragePooling2D()(pool)
    dense1 = Dense(256, activation='relu')(avg_pool)
    dense2 = Dense(128, activation='relu')(dense1)
    weights = Dense(x_train.shape[1]*x_train.shape[2]*x_train.shape[3], activation='sigmoid')(dense2)
    weights = tf.reshape(weights, (-1, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    
    input_features = input_tensor
    output = tf.matmul(input_features, weights)
    output = tf.reshape(output, [-1, x_train.shape[1], x_train.shape[2], num_classes])
    return output

main_output = main_path(input_tensor=input_layer)

# Branch path
def branch_path(input_tensor):
    flat_tensor = Flatten()(input_tensor)
    dense1 = Dense(256, activation='relu')(flat_tensor)
    dense2 = Dense(128, activation='relu')(dense1)
    output_layer = Dense(num_classes, activation='softmax')(dense2)
    return output_layer

branch_output = branch_path(input_tensor=input_layer)

# Combine outputs
combined_output = keras.layers.add([main_output, branch_output])

# Output layer
output_layer = Dense(num_classes, activation='softmax')(combined_output)

# Model construction
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

return model