# grad_cam.py (FINAL – TF 2.x SAFE)
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image


def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def make_gradcam_heatmap(img_array, model, pred_index=None):
    # 1️⃣ Get backbone (EfficientNet)
    backbone = model.layers[1]  # EfficientNetB0 (include_top=False)

    # 2️⃣ Last conv feature maps
    conv_outputs = backbone(img_array)

    with tf.GradientTape() as tape:
        tape.watch(conv_outputs)

        # forward pass through classifier head
        x = conv_outputs
        for layer in model.layers[2:]:
            x = layer(x)
        preds = x

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap, cam_path, alpha=0.4, size=(224,224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    cv2.imwrite(cam_path, overlay)
    return cam_path
