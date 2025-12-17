import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

print("--- ЛАБОРАТОРНА РОБОТА №8: TensorFlow Linear Regression ---")

num_samples = 1000
batch_size = 100
epochs = 2000  

X_data = np.random.rand(num_samples).astype(np.float32)
noise = np.random.normal(0, 1.41, num_samples).astype(np.float32)
y_data = X_data * 2 + 1 + noise

x_ph = tf.placeholder(tf.float32, shape=[None])
y_ph = tf.placeholder(tf.float32, shape=[None])

k = tf.Variable(tf.random_normal([1]), name='slope')
b = tf.Variable(tf.zeros([1]), name='bias')

y_pred = tf.multiply(x_ph, k) + b

cost = tf.reduce_sum(tf.pow(y_pred - y_ph, 2)) / (2 * batch_size)

learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

loss_history = []

with tf.Session() as sess:
    sess.run(init)
    
    print("Починаємо навчання...")
    
    for epoch in range(epochs):
        indices = np.random.choice(num_samples, batch_size)
        x_batch = X_data[indices]
        y_batch = y_data[indices]
        
        _, current_cost, current_k, current_b = sess.run(
            [optimizer, cost, k, b], 
            feed_dict={x_ph: x_batch, y_ph: y_batch}
        )
        
        loss_history.append(current_cost)
        
        if (epoch + 1) % 100 == 0:
            print(f"Епоха {epoch+1}: Cost={current_cost:.4f}, k={current_k[0]:.4f}, b={current_b[0]:.4f}")

    print("\nНавчання завершено!")
    final_k = sess.run(k)[0]
    final_b = sess.run(b)[0]
    print(f"Фінальні результати: k = {final_k:.4f}, b = {final_b:.4f}")
    print(f"Очікувані (істинні): k = 2.0000, b = 1.0000")

    plt.figure(figsize=(10, 6))
    
    plt.scatter(X_data, y_data, color='blue', alpha=0.2, label='Вхідні дані (з шумом)')
    
    y_line = final_k * X_data + final_b
    plt.plot(X_data, y_line, color='red', linewidth=3, label=f'Лінія регресії: y={final_k:.2f}x+{final_b:.2f}')
    
    plt.title("TensorFlow: Лінійна регресія")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()