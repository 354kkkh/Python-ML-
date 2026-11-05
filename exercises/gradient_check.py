import numpy as np
import pandas as pd
from pathlib import Path

h = 1e-5
def MSE(X, y, w, b):
    y_hat = X @ w + b
    error = y_hat - y
    loss = np.mean(error ** 2)
    return error, loss

def mse_plus(X, y, w, b, n=None, p_w = False, p_b = False):
    w = w.copy()  # 后续只修改本次调用的副本
    if p_w == True:
        w[n] = w[n] + h
    if p_b == True:
        b = b + h
    y_hat = X @ w + b
    error = y_hat - y
    loss = np.mean(error ** 2)
    return error, loss

def mse_minus(X, y, w, b, n=None, p_w = False, p_b = False):
    w = w.copy()  # 后续只修改本次调用的副本
    if p_w == True:
        w[n] = w[n] - h
    if p_b == True:
        b = b - h
    y_hat = X @ w + b
    error = y_hat - y
    loss = np.mean(error ** 2)
    return error, loss

def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    output_dir = ROOT_DIR / 'reports' / 'gradient_check'
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(6).reshape(3,2).astype(np.float64)
    y = np.array([7,16,29]).astype(np.float64)
    w_zero = np.zeros(2)
    b_zero = 0.0
    w_nonzer = np.array([0.5,-0.25])
    b_nonzer = 1.0

    ana_err, _ = MSE(x, y, w_zero, b_zero)
    gradient_w = (2 / 3) * x.T @ ana_err
    gradient_b = 2 * np.mean(ana_err)
    ana_err_non, _ = MSE(x, y, w_nonzer, b_nonzer)
    gradient_w_non = (2 / 3) * x.T @ ana_err_non
    gradient_b_non = 2 * np.mean(ana_err_non)

    _, w0_loss_zero_plus = mse_plus(x, y, w_zero, b_zero, n=0, p_w=True)
    _, w0_loss_zero_min = mse_minus(x, y, w_zero, b_zero, n=0, p_w=True)
    numeric_grad_w0 = (w0_loss_zero_plus - w0_loss_zero_min) / (2 * h)
    _, w1_loss_zero_plus = mse_plus(x, y, w_zero, b_zero, n=1, p_w=True)
    _, w1_loss_zero_min = mse_minus(x, y, w_zero, b_zero, n=1, p_w=True)
    numeric_grad_w1 = (w1_loss_zero_plus - w1_loss_zero_min) / (2 * h)
    _, b_loss_zero_plus = mse_plus(x, y, w_zero, b_zero, p_b=True)
    _, b_loss_zero_min = mse_minus(x, y, w_zero, b_zero, p_b=True)
    numeric_grad_b = (b_loss_zero_plus - b_loss_zero_min) / (2 * h)

    _, w0_loss_nozero_plus = mse_plus(x, y, w_nonzer, b_nonzer, n=0, p_w=True)
    _, w0_loss_nozero_min = mse_minus(x, y, w_nonzer, b_nonzer, n=0, p_w=True)
    numeric_grad_nw0 = (w0_loss_nozero_plus - w0_loss_nozero_min) / (2 * h)
    _, w1_loss_nozero_plus = mse_plus(x, y, w_nonzer, b_nonzer, n=1, p_w=True)
    _, w1_loss_nozero_min = mse_minus(x, y, w_nonzer, b_nonzer, n=1, p_w=True)
    numeric_grad_nw1 = (w1_loss_nozero_plus - w1_loss_nozero_min) / (2 * h)
    _, b_loss_nozero_plus = mse_plus(x, y, w_nonzer, b_nonzer, p_b=True)
    _, b_loss_nozero_min = mse_minus(x, y, w_nonzer, b_nonzer, p_b=True)
    numeric_grad_nb = (b_loss_nozero_plus - b_loss_nozero_min) / (2 * h)

    gra_check = pd.DataFrame({
        'point': range(1,7),
        'parameter': ['w0_zero', 'w1_zero', 'b_zero', 'w0_nonzero', 'w1_nonzero', 'b_nonzero'],
        'analytic': [gradient_w[0], gradient_w[1], gradient_b, gradient_w_non[0], gradient_w_non[1], gradient_b_non],
        'numeric': [numeric_grad_w0, numeric_grad_w1, numeric_grad_b, numeric_grad_nw0, numeric_grad_nw1, numeric_grad_nb],
        'abs_error': [abs(gradient_w[0]-numeric_grad_w0), abs(gradient_w[1]-numeric_grad_w1), abs(gradient_b-numeric_grad_b), abs(gradient_w_non[0]-numeric_grad_nw0), abs(gradient_w_non[1]-numeric_grad_nw1), abs(gradient_b_non-numeric_grad_nb)],
        'passed': [np.isclose(gradient_w[0], numeric_grad_w0, atol=1e-6,rtol=1e-5), np.isclose(gradient_w[1], numeric_grad_w1, atol=1e-6,rtol=1e-5), np.isclose(gradient_b, numeric_grad_b, atol=1e-6,rtol=1e-5), np.isclose(gradient_w_non[0], numeric_grad_nw0, atol=1e-6,rtol=1e-5), np.isclose(gradient_w_non[1], numeric_grad_nw1, atol=1e-6,rtol=1e-5), np.isclose(gradient_b_non, numeric_grad_nb, atol=1e-6,rtol=1e-5)]
    })
    gra_check.to_csv(output_dir / 'gradient_check.csv', index=False)
    print(gra_check)


if __name__ == "__main__":
    main()