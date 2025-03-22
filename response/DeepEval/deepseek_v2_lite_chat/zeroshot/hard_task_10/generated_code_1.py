from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, concatenate, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def dl_model():
    # Input layers
    inputs = [input_branch, input_branch_branch]

    # Feature extraction paths
    fe_path1 = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs[0])
    fe_path2 = Conv2D(64, (1, 7), padding='valid')(inputs[0])
    fe_path2 = Conv2D(64, (7, 1), padding='valid')(fe_path2)
    concat = Add()([fe_path1, fe_path2])

    # Align the output dimensions with the input image's channel
    conv1x1_2 = Conv2D(64, (1, 1), activation='relu', padding='same')(concat)

    # Main path output after addition and 1x1 convolution
    main_path = conv1x1_2

    # Branch: directly connect the input
    branch = inputs[1]

    # Merging the outputs of the main path and the branch
    merged = concatenate([main_path, branch])

    # Classification layers
    flatten = Flatten()(merged)
    outputs = Dense(128, activation='relu')(flatten)
    outputs = Dense(10, activation='softmax')(outputs)  # Assuming 10 classes

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Load and prepare the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)