import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def method():
    # Example: Load an image
    # img_path = 'path_to_your_image.jpg'  # Replace with your image path

    # 修改为本地数据文件
    img_path = 'evaluation/dynamic_checking/baselines/MLEval/deepseek_coder_v2_lite_instruct/testcases/task_5.jpg'

    img = image.load_img(img_path, target_size=(224, 224))  # Resize to fit the model's input size
    img_array = image.img_to_array(img)  # Convert the image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Assuming you have a pre-trained model that expects RGB input
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)

    # Modify the input layer to accept RGB images
    input_layer = base_model.layers[0].input
    modified_input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
    modified_model = tf.keras.models.Model(inputs=[input_layer], outputs=base_model.layers[1:](modified_input_layer))

    # Predict using the modified model
    predictions = modified_model.predict(img_array)
    decoded_predictions = tf.keras.applications.vgg16.decode_predictions(predictions, top=5)

    # Prepare the output
    output = []
    for (_, label, prob) in decoded_predictions[0]:
        output.append((label, prob))

    return output

# Call the method for validation
print(method())