import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    main_conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch path
    branch_conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_conv1_3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(branch_conv1_1)
    branch_conv3_1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(branch_conv1_3)
    
    # Concatenate the outputs of the main path and branch path
    concat = Concatenate()([main_conv1, branch_conv3_1])
    
    # 1x1 convolution after concatenation to maintain the same channel dimensions as input
    main_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Add input directly to the output of the main path for fusion
    fused_output = Add()([input_layer, main_output])
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Train the model (example with small epoch number for demonstration purposes)
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_data=(x_test, y_test))