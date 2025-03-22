import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    input_layer = Input(shape=(32, 32, 3))
    
    # First branch
    def block(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=256, activation='relu')(avg_pool)
        dense2 = Dense(units=128, activation='relu')(dense1)
        branch_output = Dense(units=10, activation='softmax')(dense2)
        
        return branch_output
    
    branch1 = block(input_tensor=input_layer)

    # Second branch
    def block(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=256, activation='relu')(avg_pool)
        dense2 = Dense(units=128, activation='relu')(dense1)
        branch_output = Dense(units=10, activation='softmax')(dense2)
        
        return branch_output
    
    branch2 = block(input_tensor=input_layer)

    # Concatenate outputs from both branches
    concat_layer = Concatenate()([branch1, branch2])
    flatten_layer = Flatten()(concat_layer)
    
    # Fully connected layer
    dense3 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')