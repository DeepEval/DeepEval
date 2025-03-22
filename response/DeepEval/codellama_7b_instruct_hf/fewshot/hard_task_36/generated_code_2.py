from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.applications import VGG16

# Load the VGG16 model as a pre-trained model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in vgg_model.layers:
    layer.trainable = False

 和 return model
def dl_model(): 
    # Define the input shape
    input_shape = (224, 224, 3)

    # Define the main pathway
    main_path = Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)(vgg_model.output)
    main_path = Conv2D(64, (1, 1), activation='relu')(main_path)
    main_path = Conv2D(64, (1, 1), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)
    main_path = Dropout(0.5)(main_path)

    # Define the branch pathway
    branch_path = Conv2D(64, (3, 3), activation='relu', input_shape=input_shape)(vgg_model.output)
    branch_path = Conv2D(64, (1, 1), activation='relu')(branch_path)
    branch_path = Conv2D(64, (1, 1), activation='relu')(branch_path)

    # Define the output layer
    output_layer = GlobalAveragePooling2D()(main_path)
    output_layer = Flatten()(output_layer)
    output_layer = Dense(10, activation='softmax')(output_layer)

    # Create the model
    model = Model(inputs=vgg_model.input, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model