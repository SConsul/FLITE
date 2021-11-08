# Functions to show checkpoint log plots

from parse_logs import plot_train_log

# Show baseline plots
def show_baseline_step():
    name = "Baseline"
    train_log_path = "./experiment_results/checkpoint_outputs/orbit_baseline/train-2021-11-02-22-07-56/log.txt"

    # Create and show plot from log.txt
    plot_train_log(train_log_path, name)

def show_baseline_epoch():
    name = "Baseline"
    train_log_path = "./experiment_results/checkpoint_outputs/orbit_baseline/train-2021-11-02-22-07-56/log.txt"

    # Create and show plot from log.txt
    plot_train_log(train_log_path, name, by_step=False)

# Show blur heuristic plots
def show_blur_step():
    name = "Blur Heuristic"
    train_log_path = "./experiment_results/checkpoint_outputs/blur_heuristic/train-2021-11-08-04-11-54/log.txt"

    # Create and show plot from log.txt
    plot_train_log(train_log_path, name)

def show_blur_epoch():
    name = "Blur Heuristic"
    train_log_path = "./experiment_results/checkpoint_outputs/blur_heuristic/train-2021-11-08-04-11-54/log.txt"

    # Create and show plot from log.txt
    plot_train_log(train_log_path, name, by_step=False)

show_baseline_step()
show_baseline_epoch()
show_blur_step()
show_blur_epoch()