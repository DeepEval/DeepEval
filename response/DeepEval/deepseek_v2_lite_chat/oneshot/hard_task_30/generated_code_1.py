import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, Add, Lambda, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.layers import Concatenate, BatchNormalization, Activation

def dl_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Set up the input shape
    input_shape = (32, 32, 3)
    
    # Create the input layer
    input_layer = Input(shape=input_shape)
    
    # First block: dual-path structure
    def first_block(input_tensor):
        # Main path
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        
        # Branch path
        branch_tensor = input_tensor
        # Concatenate the main path and branch path
        concat_tensor = Concatenate()([conv2, branch_tensor])
        
        return concat_tensor
    
    block1_output = first_block(input_tensor=input_layer)
    batch_norm1 = BatchNormalization()(block1_output)
    activation1 = Activation('relu')(batch_norm1)
    
    # Second block: three groups of depthwise separable convolutions
    def second_block(input_tensor):
        # Split the tensor into three groups
        split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=[16, 16, 16], axis=1))(input_tensor)
        
        # Paths for each group
        path1 = Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')(split1)
        path2 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(split2)
        path3 = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(split3)
        
        # Concatenate the paths
        concatenated_tensor = Concatenate()([path1, path2, path3])
        
        return concatenated_tensor
    
    block2_output = second_block(activation1)
    batch_norm2 = BatchNormalization()(block2_output)
    activation2 = Activation('relu')(batch_norm2)
    
    # Flatten and fully connected layers
    flatten = Flatten()(activation2)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)