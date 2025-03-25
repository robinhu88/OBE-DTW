import numpy as np
import matplotlib.pyplot as plt
import os


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

    xy_coords = np.array([item[1:3] for item in parsed_data])  # 只提取xy坐标

    mag_coords1 = np.array([item[3] for item in parsed_data[:2000]])
    mag_coords2 = np.array([item[4] for item in parsed_data[:2000]])
    mag_coords3 = np.array([item[5] for item in parsed_data[:2000]])

    # 创建一个包含三个子图的图形
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # 绘制第一个磁坐标通道
    axs[0].plot(mag_coords1, color='b', label='Mag Coord 1')
    axs[0].set_title('Magnetic Coordinate 1')
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('Value')
    axs[0].legend()
    axs[0].grid(True)

    # 绘制第二个磁坐标通道
    axs[1].plot(mag_coords2, color='g', label='Mag Coord 2')
    axs[1].set_title('Magnetic Coordinate 2')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('Value')
    axs[1].legend()
    axs[1].grid(True)

    # 绘制第三个磁坐标通道
    axs[2].plot(mag_coords3, color='r', label='Mag Coord 3')
    axs[2].set_title('Magnetic Coordinate 3')
    axs[2].set_xlabel('Sample Index')
    axs[2].set_ylabel('Value')
    axs[2].legend()
    axs[2].grid(True)

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

    return xy_coords


def calculate_distances_and_find_first_decrease_after_10(points, start, max_points=20000):
    distances = []
    first_decrease_index = None
    previous_distance = float('inf')

    end_point = min(start + max_points, len(points))  # 确保结束点不超过数据长度
    for i in range(start + 1, end_point + 1):
        if i <= len(points):
            distance = np.linalg.norm(points[i - 1] - points[start])
            distances.append(distance)
            if i > start + 10 and distance < previous_distance and first_decrease_index is None:
                first_decrease_index = i - start - 1  # 记录相对于起点的索引距离
            previous_distance = distance

    return distances, first_decrease_index


def plot_distances(distances_list, title="", output_dir="."):
    fig, ax = plt.subplots(figsize=(14, 7))

    for idx, (start, distances, first_decrease_index) in enumerate(distances_list):
        x = np.arange(start + 2, start + len(distances) + 2)  # 起始点为start+2，因为是从第start+1个点到第i个点的距离
        ax.plot(x, distances, label=f'Start from point {start + 1}')
        if first_decrease_index is not None and first_decrease_index >= 9:  # 从第10个点后开始标记
            relative_distance = first_decrease_index + 1  # 相对距离是索引加1
            ax.annotate(f'First Decrease\nIndex Distance: {relative_distance}',
                        (x[first_decrease_index], distances[first_decrease_index]),
                        textcoords="offset points", xytext=(0, 10), ha='center',
                        arrowprops=dict(facecolor='red', shrink=0.05))

    plt.legend()
    plt.xlabel('Point Index')
    plt.ylabel('Distance')
    plt.title(title)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(output_file)
    plt.close(fig)


if __name__ == "__main__":
    map_file = r'C:\Users\Administrator\Desktop\data\2map155531.txt'
    # test_files = [
    #     r'C:\Users\Administrator\Desktop\data\data153344.txt',
    #     r'C:\Users\Administrator\Desktop\data\data154324.txt',
    #     r'C:\Users\Administrator\Desktop\data\data161152.txt',
    #     r'C:\Users\Administrator\Desktop\data\data162030.txt',
    #     r'C:\Users\Administrator\Desktop\data\data163356.txt',
    #     r'C:\Users\Administrator\Desktop\data\data164715.txt',
    #     r'C:\Users\Administrator\Desktop\data\data165807.txt',
    #     r'C:\Users\Administrator\Desktop\data\data171421.txt',
    #     r'C:\Users\Administrator\Desktop\data\data172624.txt'
    # ]
    test_files = [
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata153344.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata154324.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata161152.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata155531.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata162030.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata163356.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata164715.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata165807.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata171421.txt',
        r'C:\Users\Administrator\Desktop\data100Hz\100hzdata172624.txt'
    ]

    step_starts = [0, 49, 99, 149, 199, 200, 250]  # 起始点索引
    max_points = 20000  # 每次计算的最大点数

    for test_file in test_files:
        xy_coords = read_data(test_file)

        distances_list = []

        for start in step_starts:
            if start < len(xy_coords):  # 确保起始点在数据范围内
                distances, first_decrease_index = calculate_distances_and_find_first_decrease_after_10(xy_coords, start,
                                                                                                       max_points)
                distances_list.append((start, distances, first_decrease_index))

                if first_decrease_index is not None and first_decrease_index >= 9:
                    print(
                        f"File: {os.path.basename(test_file)}, Start from point {start + 1}, First decrease at index {first_decrease_index} (point {start + first_decrease_index + 2}), Relative Distance: {first_decrease_index + 1}")
                else:
                    print(
                        f"File: {os.path.basename(test_file)}, Start from point {start + 1}, No decrease found after the 10th point.")

        plot_distances(distances_list, title=f"Distances for {os.path.basename(test_file)}", output_dir="output_images")