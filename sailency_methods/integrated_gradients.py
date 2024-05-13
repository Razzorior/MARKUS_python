import tensorflow as tf

def integrated_gradients(model, baseline, image_sample, steps=50):
    alphas = tf.linspace(0.0, 1.0, steps)

    def interpolate_images(baseline, image_sample, alpha):
        return baseline + alpha * (image_sample - baseline)

    imgs = [interpolate_images(baseline, image_sample, alpha) for alpha in alphas]
    imgs = tf.concat(imgs, axis=0)

    with tf.GradientTape() as tape:
        tape.watch(imgs)
        logits = model(imgs)

    grads = tape.gradient(logits, imgs)
    avg_grads = tf.reduce_mean(grads, axis=0)

    integrated_grad = (image_sample - baseline) * avg_grads
    integrated_grad = tf.reduce_mean(integrated_grad, axis=0)  # Average along the steps dimension
    return integrated_grad