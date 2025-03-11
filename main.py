import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os
import pandas as pd


class ObeDtwCalculator:
    def __init__(self):
        pass

    def d(self, x, y):
        return np.linalg.norm(x - y)

    def obe_dtw_distance(self, s1, s2):
        ts_a = s1.T
        ts_b = s2.T
        M, N = len(ts_a), len(ts_b)

        dist_matrix = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                dist_matrix[i, j] = self.d(ts_a[i], ts_b[j])

        cum_matrix = np.full((M + 1, N), np.inf)
        cum_matrix[0, :] = 0

        for i in range(1, cum_matrix.shape[0]):
            for j in range(cum_matrix.shape[1]):
                cost = dist_matrix[i - 1, j]
                min_cost = min(float(cum_matrix[i - 1, j]),
                               float(cum_matrix[i - 1, max(j - 1, 0)]),
                               float(cum_matrix[i - 1, max(j - 2, 0)]))
                cum_matrix[i, j] = cost + min_cost

        best_end_point_index = np.argmin(cum_matrix[-1, :])
        return best_end_point_index


def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data_lines = [line for line in lines if not line.startswith('GPSsecond')]
    parsed_data = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) != 6:
            print(f"Warning: Line does not contain 6 values: {line}")
            continue
        try:
            parsed_data.append([float(part) for part in parts])
        except ValueError as e:
            print(f"Error parsing line: {line}. Error: {e}")
            continue

    mag_data = np.array([item[3:6] for item in parsed_data])
    xy_coords = np.array([item[1:3] for item in parsed_data])

    return mag_data, xy_coords


def create_windows(data, window_size=12):
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        windows.append(window)
        # if i % 100 == 0:
        #     print(f"Created {len(windows)} windows so far...")
    return windows


def find_best_match(args):
    target_window, base_mag, base_xy_coordinates, dtw_calculator = args

    # 直接使用整个基准序列进行比较
    best_end_point_index = dtw_calculator.obe_dtw_distance(np.array(target_window).reshape(-1, 3).T,
                                                           np.array(base_mag).reshape(-1, 3).T)

    # 使用best_end_point_index来获取基准序列中的最佳匹配点坐标
    best_match_point = base_xy_coordinates[best_end_point_index]

    return best_match_point, best_end_point_index


def plot_sequences(matches, test_end_points, title="", output_dir="."):
    fig, ax = plt.subplots()

    match_points = [point for point, _ in matches]

    if len(test_end_points) > 1:
        x_coords, y_coords = zip(*test_end_points)
        ax.plot(x_coords, y_coords, '-o', label='Real Window', color='blue', markersize=0.5)

    if len(match_points) > 1:
        x_match, y_match = zip(*match_points)
        ax.plot(x_match, y_match, '-s', label='Match Points', color='red', markersize=2)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_file)
    plt.close(fig)
    plt.show()

def calculate_errors(errors):
    mae = np.mean(errors)
    mse = np.mean(np.square(errors))
    rmse = np.sqrt(mse)

    return mae, mse, rmse
if __name__ == "__main__":
    map_file = r'C:\Users\Administrator\Desktop\data\2map155531.txt'
    test_files = [ r'C:\Users\Administrator\Desktop\data\data153344.txt',
        r'C:\Users\Administrator\Desktop\data\data154324.txt',
        r'C:\Users\Administrator\Desktop\data\data161152.txt',
        r'C:\Users\Administrator\Desktop\data\data162030.txt',
        r'C:\Users\Administrator\Desktop\data\data163356.txt',
        r'C:\Users\Administrator\Desktop\data\data164715.txt',
        r'C:\Users\Administrator\Desktop\data\data165807.txt',
        r'C:\Users\Administrator\Desktop\data\data171421.txt',
        r'C:\Users\Administrator\Desktop\data\data172624.txt']

    # 读取基准地图文件
    map_mag, map_xy = read_data(map_file)

    dtw_calculator = ObeDtwCalculator()

    for test_file in test_files:
        all_test_xy_coordinates = []
        all_matches = []

        test_mag, test_xy = read_data(test_file)
        test_mag_windows = create_windows(test_mag)
        test_xy_windows = create_windows(test_xy)

        # 收集每个窗口的最后一个坐标
        for window in test_xy_windows:
            all_test_xy_coordinates.append(window[-1])

        pool_args = [(tw, map_mag, map_xy, dtw_calculator) for tw in test_mag_windows]
        with Pool(cpu_count()) as p:
            results = p.map(find_best_match, pool_args)

        matches = []
        errors = []
        for result, test_coord in zip(results, all_test_xy_coordinates):
            best_match_point, best_match_index = result
            #print(f"Test Coordinate: {test_coord}, Best Match Point: {best_match_point}")
            error = np.linalg.norm(np.array(test_coord) - np.array(best_match_point))
            # 输出误差值以便检查
            #print(f"Calculated Error: {error}")
            matches.append((best_match_point, error))
            errors.append(error)

        avg_mae, avg_mse, avg_rmse = calculate_errors(errors)

        # 输出误差
        print(f"File: {test_file}, MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}")
        all_matches.extend(matches)



        # 输出每个测试文件的最佳匹配点和误差
        # for idx, (point, error) in enumerate(matches):
        #     print(f"Test file: {test_file}, Window Index: {idx}, Best Match Point: {point}, Error: {error}")

        # 绘制当前测试文件的结果

        plot_sequences(matches, all_test_xy_coordinates, title=f"Matches for {os.path.basename(test_file)}",
                       output_dir="output_images_12")



