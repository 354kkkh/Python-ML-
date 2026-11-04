import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

def linear_regression(X, y):
    reg = LinearRegression(fit_intercept=True).fit(X, y)
    pre = reg.predict(X)
    mse = ((pre - y) ** 2).mean()
    return pre, mse

def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    output_dir = ROOT_DIR / 'reports' / 'gradient_descent'
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(6).reshape(3,2)
    y = np.array([7,16,29])
    w = np.zeros(2)
    b = 0.0

    n = len(y)
    y_hat = x @ w + b
    error = y_hat - y
    loss = np.mean(error**2)
    learning_rate = 0.01

    step = [0]
    mse = [loss]
    for i in range(1,501):
        gradient_w = (2 / n) * x.T @ error
        gradient_b = 2 * np.mean(error)

        w_new = w - learning_rate * gradient_w
        b_new = b - learning_rate * gradient_b
        prediction_new = x @ w_new + b_new
        loss_after = np.mean((prediction_new - y) ** 2)

        if i == 500:
            print('w:',w_new, 'b:', b_new)
            print(prediction_new)

        w = w_new
        b = b_new
        error = prediction_new - y
        step.append(i)
        mse.append(loss_after)

    print('初始MSE:', mse[0], '第1步MSE:', mse[1], '\n第10步MSE:', mse[10], '\n第100步MSE:', mse[100], '\n第500步MSE:', mse[500])

    loss_history = pd.DataFrame({'step': step, 'mse': mse})
    loss_history.to_csv(output_dir / 'loss_history.csv', index=False)

    lin_pre, lin_mse = linear_regression(x, y)
    pre_compar = pd.DataFrame({
        'sample_index': range(3),
        'y_true': y,
        'numpy_pred': prediction_new,
        'sklearn_pred': lin_pre
    })
    metr = pd.DataFrame({
        'model': ['NumPy_GD', 'LinearRegression'],
        'n_samples': 3,
        'mse': [mse[500], lin_mse],
    })
    pre_compar.to_csv(output_dir / 'prediction_comparison.csv', index=False)
    metr.to_csv(output_dir / 'metrics.csv', index=False)
    print(pre_compar, '\n', metr)


if __name__ == '__main__':
    main()
