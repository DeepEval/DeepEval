from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate
from keras.layers import Layer

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Function to create the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Global Average Pooling
    block1 = GlobalAveragePooling2D()(input_layer)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(block1)
    dense2 = Dense(64, activation='relu')(dense1)

    # Reshape for multiplication with input
    dense1_shape = Dense.compute_output_shape(dense1)
    dense2_shape = Dense.compute_output_shape(dense2)

    # Weighted feature output
    weight_feature = Dense(x_train.shape[1], kernel_initializer='zeros')(block1)
    weight_feature = Reshape(x_train.shape)(weight_feature)
    weighted_feature = multiply([weight_feature, input_layer])

    # Block 2: Convolutional layers and max pooling
    conv1 = Conv2D(64, kernel_size=3, activation='relu')(input_layer)
    conv2 = Conv2D(64, kernel_size=3, activation='relu')(conv1)
    pool = MaxPooling2D()(conv2)

    # Branch from Block 1 to Block 2
    branch = Dense(pool.shape[1], name='branch_dense')(block1)
    branch = Reshape(pool.shape)(branch)
    branch = multiply([branch, pool])

    # Concatenate the main path and the branch
    fused_output = Concatenate()([weighted_feature, branch])

    # Fully connected layers for classification
    output = Dense(10, activation='softmax')(fused_output)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and return the model
model = dl_model()
model.summary()