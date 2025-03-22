import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.datasets import cifar10

def dl_model():     
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Preprocess data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    input_layer = Input(shape=(32, 32, 3))
    
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    max_pooling_1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv)
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    max_pooling_3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)
    
    concat_poolings = Concatenate()([max_pooling_1, max_pooling_2, max_pooling_3])
    
    bath_norm = BatchNormalization()(concat_poolings)
    
    flatten_layer = Flatten()(bath_norm)
    
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
# model.fit(x_train, y_train, batch_size=32, epochs=10)

# Evaluate model
# loss, accuracy = model.evaluate(x_test, y_test)
# print('Test accuracy:', accuracy)