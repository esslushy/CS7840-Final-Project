import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def gaussian_mi(x: torch.Tensor, y: torch.Tensor):
    """
    Computes MI assuming (x,y) are jointly Gaussian:
    I = -1/2 log(1 - rho^2)
    """
    x = x.squeeze()
    y = y.squeeze()

    rho = torch.corrcoef(torch.stack([x, y]))[0, 1]
    return -0.5 * torch.log(1 - rho**2 + 1e-12) # https://math.nyu.edu/~kleeman/infolect7.pdf

class Net(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

def run_experiment(steps=1000):

    torch.manual_seed(0)

    model = Net()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    mi_history = []
    eq_error_history = []

    for step in range(steps):

        x = torch.randn(2048, 1)

        fx = model(x)
        fnegx = model(-x)

        # Equivariance loss
        loss = torch.mean((fx + fnegx)**2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # ---- Logging ----
        if step % 20 == 0:
            with torch.no_grad():
                mi = gaussian_mi(fx, fnegx)
                eq_error = torch.mean((fx + fnegx)**2).item()

                mi_history.append(mi)
                eq_error_history.append(eq_error)

    return mi_history, eq_error_history

def plot_results(mi_history, eq_error_history):

    fig, ax1 = plt.subplots()

    # Plot MI curve
    line1, = ax1.plot(mi_history)
    

    ax1.set_xlabel("Training (x20 steps)")
    ax1.set_ylabel("Mutual Information")
    ax1.set_title("MI increases as equivariance is learned")

    # Second axis for equivariance error
    ax2 = ax1.twinx()
    line2, = ax2.plot(eq_error_history, color="orange")
    ax2.set_ylabel("Equivariance Error")

    # Combined legend
    ax1.legend(
        [line1, line2],
        ["Mutual Information", "Equivariance Error", "Total Information (Max)"],
        loc="best"
    )

    plt.savefig("test.pdf")

if __name__ == "__main__":

    mi_hist, eq_hist = run_experiment()
    plot_results(mi_hist, eq_hist)