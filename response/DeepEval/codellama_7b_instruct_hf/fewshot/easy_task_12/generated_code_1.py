from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, SeparableConv2D, BatchNormalization

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(SeparableConv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Define the branch path
    branch_path = Sequential()
    branch_path.add(Conv2D(64, (5, 5), padding='same'))
    branch_path.add(Activation('relu'))
    branch_path.add(MaxPooling2D(pool_size=(2, 2)))

    # Merge the main and branch paths
    merged_path = Sequential()
    merged_path.add(Add()([model.output, branch_path.output]))

    # Flatten the output
    merged_path.add(Flatten())

    # Add a fully connected layer
    merged_path.add(Dense(128, activation='relu'))
    merged_path.add(Dense(64, activation='relu'))
    merged_path.add(Dense(10, activation='softmax'))

    # Define the model
    model = Sequential([model, branch_path, merged_path])

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
                
    
    return model