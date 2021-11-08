# Functions to create charts and graphs from log output. 

import matplotlib.pyplot as plt

def create_graph(title, xPoints, yPoints, xLabel="", yLabel=""):
    # Plot graph
    plt.plot(xPoints, yPoints)
    plt.title(title)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.show()


def plot_train_log(path_to_log_file, name="Train", by_step=True):
    lines = []
    with open(path_to_log_file) as f:
        lines = f.readlines()

    train_losses = []
    frame_accuracies = []
    step_count = 0
    for line in lines:
        # Plot by step
        if by_step:
            # Format: epoch [1/10][1/2200], train loss: 27.4976845, frame_acc: 100.00, time/task: 0m03s

            if line.startswith("epoch"):
                # Split on colon and get train loss and frame accuracy values
                values = line.split(',')
                if (len(values) == 4):
                    train_loss = float(values[1].strip().split(':')[1])
                    frame_acc = float(values[2].strip().split(':')[1])/100.0

                    train_losses.append(train_loss)
                    frame_accuracies.append(frame_acc)

                    step_count += 1
        # Plot by epoch
        else:
            # Format: epoch [1/10] train loss: 9.1337709 frame_acc: 80.26 (0.43)  time/epoch: 134m35s
            if line.startswith("epoch") and ',' not in line:
                # Split space and get train loss and frame accuracy values
                values = line.split()

                train_loss = float(values[4])
                frame_acc = float(values[6])/100.0

                train_losses.append(train_loss)
                frame_accuracies.append(frame_acc)

                step_count += 1


    steps = list(range(step_count))

    xLabel = "step"
    if not by_step:
        xLabel = "epoch"

    # Create graphs for train and eval
    create_graph(name + ": loss/train", steps, train_losses, xLabel, "loss")
    create_graph(name + ": frame_acc/eval", steps, frame_accuracies, xLabel, "accuracy")


def parse_test_log(path_to_log_file):
    pass
