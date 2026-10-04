import numpy as np

b = 5
x = np.arange(6).reshape(3,2)
w = np.array([2,3])
y = np.array([7,16,29])

y_h = x @ w + b
err = (y - y_h)**2
MSE = np.mean(err)

print('y_hat:', y_h, 'MSE:', MSE)
print('\nx:', x.shape, '\ny:', y.shape, '\nw:', w.shape,
      '\ny_hat:', y_h.shape, '\nerr:', err.shape,)
print('\nx_mean_axis=0:', np.mean(x, axis=0),
      '\nx_mean_axis=1:', np.mean(x, axis=1))    #axis=0按行求均值，axis=1按列求均值


