import numpy as np

b = 5
x = np.arange(6).reshape(3,2)
w = np.array([2,3])
y = np.array([7,16,29])
y_reshape = y.reshape(3,1)

y_h = x @ w + b
err = (y - y_h)**2
MSE = np.mean(err)

print('y_hat:', y_h, 'MSE:', MSE)
print('\nx:', x.shape, '\ny:', y.shape, '\nw:', w.shape,
      '\ny_hat:', y_h.shape, '\nerr:', err.shape,)
print('\nx_mean_axis=0:', np.mean(x, axis=0),
      '\nx_mean_axis=1:', np.mean(x, axis=1))    #axis=0得到每列均值，axis=1得到每行均值
print('\nx*w:', x * w,
      '\nx@w:', x @ w,
      '\ny_reshape-y_hat:', y_reshape-y_h,
      '\nE[(y_reshape-y_hat)^2]:', np.mean((y_reshape-y_h)**2),
      '\nE[(y_reshape-y_hat_resh)^2]:', np.mean((y_reshape-y_h.reshape(3,1))**2))

'''
梯度下降训练
'''
# 初始参数：使用浮点数
w_g = np.zeros(2)
b_g = 0.0

learning_rate = 0.01
n = len(y)

# ① 用旧参数预测，计算损失
prediction = x @ w_g + b_g
error = prediction - y
loss_before = np.mean(error ** 2)

# ② 两个梯度都由同一组旧参数计算
gradient_w = (2 / n) * x.T @ error
gradient_b = 2 * np.mean(error)

# ③ 沿负梯度方向更新参数
w_g_new = w_g - learning_rate * gradient_w
b_g_new = b_g - learning_rate * gradient_b

# ④ 用新参数重新预测，检查损失
prediction_new = x @ w_g_new + b_g_new
loss_after = np.mean((prediction_new - y) ** 2)

print("更新前参数：", w_g, b_g)
print("更新前预测：", prediction)
print("更新前MSE：", loss_before)

print("权重梯度：", gradient_w)
print("偏置梯度：", gradient_b)

print("更新后参数：", w_g_new, b_g_new)
print("更新后预测：", prediction_new)
print("更新后MSE：", loss_after)