import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file')
    args = parser.parse_args()

    df = pd.read_json(args.log_file)

    plt.figure(figsize=(8, 4))
    plt.plot(df['iteration'], df['main/loss'], linewidth=1, label='train')
    plt.plot(df['iteration'][df['val/loss'].notnull()], df['val/loss'][df['val/loss'].notnull()], linewidth=1, label='validation')
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    dir_ = '/'.join(args.log_file.split('/')[:-1])
    plt.savefig(os.path.join(dir_, 'loss_history.png'))
    # plt.show()
