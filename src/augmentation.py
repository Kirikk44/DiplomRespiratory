# augmentation.py (ИСПРАВЛЕННАЯ ВЕРСИЯ)

"""
Аугментация NTU RGB+D скелетных данных
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import json

# =============================================
# ЧТЕНИЕ .SKELETON ФАЙЛОВ
# =============================================

def read_skeleton_file(file_path):
    """
    Читает .skeleton файл NTU RGB+D
    
    Возвращает numpy array [T, V, C] где:
        T = frames (количество кадров)
        V = 25 joints (суставы)
        C = 3 (x, y, z координаты)
    """
    with open(file_path, 'r') as f:
        framecount = int(f.readline())
        frames_data = []
        
        for t in range(framecount):
            bodycount = int(f.readline())
            
            if bodycount == 0:
                # Нет тела - добавляем нули
                frames_data.append(np.zeros((25, 3)))
                continue
            
            # Читаем информацию о теле (пропускаем)
            body_info_line = f.readline()
            
            # Количество суставов
            jointcount = int(f.readline())
            
            # Читаем координаты суставов
            joints = []
            for v in range(jointcount):
                joint_line = f.readline().split()
                # Первые 3 значения - это x, y, z
                x = float(joint_line[0])
                y = float(joint_line[1])
                z = float(joint_line[2])
                joints.append([x, y, z])
            
            # Если есть дополнительные тела (второй человек), пропускаем
            for m in range(1, bodycount):
                f.readline()  # body info
                jc = int(f.readline())
                for v in range(jc):
                    f.readline()
            
            frames_data.append(np.array(joints))
        
        skeleton = np.array(frames_data)
        return skeleton


def load_skeleton_dataset(dataset_path, split='train', class_name='cough'):
    """
    Загружает все .skeleton файлы из папки
    
    Args:
        dataset_path: путь к ntu_cough_dataset
        split: 'train', 'val', или 'test'
        class_name: 'cough' или 'non_cough'
    
    Returns:
        data: list of [T, V, 3] numpy arrays
        filenames: list of Path объектов
    """
    dataset_path = Path(dataset_path)
    class_dir = dataset_path / split / class_name
    
    if not class_dir.exists():
        raise ValueError(f"Папка не найдена: {class_dir}")
    
    skeleton_files = sorted(class_dir.glob('*.skeleton'))
    
    if len(skeleton_files) == 0:
        raise ValueError(f"Не найдено .skeleton файлов в {class_dir}")
    
    print(f"Найдено {len(skeleton_files)} файлов в {class_dir}")
    
    data = []
    filenames = []
    
    for skel_file in tqdm(skeleton_files, desc=f"Loading {split}/{class_name}"):
        try:
            skeleton = read_skeleton_file(skel_file)
            data.append(skeleton)
            filenames.append(skel_file)
        except Exception as e:
            print(f"⚠️  Ошибка в {skel_file.name}: {e}")
            continue
    
    return data, filenames


def parse_ntu_filename(filename):
    """
    Парсит имя файла NTU RGB+D
    
    Формат: SsssCcccPpppRrrrAaaa.skeleton
    Пример: S001C002P003R001A041.skeleton
    
    Returns:
        dict с полями: setup, camera, performer, replication, action
    """
    name = Path(filename).stem  # Убираем .skeleton
    
    info = {
        'setup': int(name[1:4]),       # S001 -> 1
        'camera': int(name[5:8]),      # C002 -> 2
        'performer': int(name[9:12]),  # P003 -> 3
        'replication': int(name[13:16]), # R001 -> 1
        'action': int(name[17:20])     # A041 -> 41
    }
    
    return info


# =============================================
# МАТЕМАТИЧЕСКИЕ ФОРМУЛЫ ТРАНСФОРМАЦИЙ
# =============================================

def rotation_matrix_y(angle_degrees: float) -> np.ndarray:
    """Матрица вращения вокруг оси Y (вертикальная ось)"""
    theta = np.radians(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    R = np.array([
        [cos_t,  0.0,  sin_t],
        [0.0,    1.0,  0.0],
        [-sin_t, 0.0,  cos_t]
    ], dtype=np.float32)
    
    return R


def flip_matrix_x() -> np.ndarray:
    """Матрица горизонтального отражения (flip по оси X)"""
    F = np.array([
        [-1.0, 0.0, 0.0],
        [0.0,  1.0, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=np.float32)
    
    return F


def rotation_matrix_x(angle_degrees: float) -> np.ndarray:
    """Вращение вокруг оси X (наклон камеры вверх/вниз)"""
    theta = np.radians(angle_degrees)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    R = np.array([
        [1.0,  0.0,    0.0],
        [0.0,  cos_t, -sin_t],
        [0.0,  sin_t,  cos_t]
    ], dtype=np.float32)
    
    return R


# =============================================
# ПРИМЕНЕНИЕ ТРАНСФОРМАЦИЙ
# =============================================

def apply_transform_to_skeleton(seq: np.ndarray,
                                R: np.ndarray,
                                center_joint: int = 0) -> np.ndarray:
    """
    Применяет трансформацию к скелетной последовательности
    
    Args:
        seq: [T, V, 3] - T кадров, V суставов, (x,y,z)
        R: [3, 3] - матрица трансформации
        center_joint: индекс центрального сустава (0 = spine base)
    
    Returns:
        out: [T, V, 3] - трансформированная последовательность
    """
    assert seq.ndim == 3 and seq.shape[2] == 3, "Ожидается [T, V, 3]"
    
    T, V, C = seq.shape
    out = np.empty_like(seq, dtype=np.float32)
    
    # Координаты центрального сустава
    center = seq[:, center_joint, :]  # [T, 3]
    
    for t in range(T):
        pts = seq[t]               # [V, 3]
        c = center[t:t+1, :]       # [1, 3]
        
        # Центрируем, трансформируем, возвращаем
        pts_centered = pts - c
        pts_transformed = pts_centered @ R.T
        pts_new = pts_transformed + c
        
        out[t] = pts_new
    
    return out


# =============================================
# ВИРТУАЛЬНЫЕ КАМЕРЫ (БЕЗ ДУБЛИРОВАНИЯ NTU)
# =============================================

class VirtualCamera:
    """Конфигурация виртуальной камеры"""
    
    def __init__(self, 
                 name: str,
                 rotation_y: float = 0.0,
                 rotation_x: float = 0.0,
                 flip_horizontal: bool = False):
        self.name = name
        self.rotation_y = rotation_y
        self.rotation_x = rotation_x
        self.flip_horizontal = flip_horizontal
    
    def get_transform_matrix(self) -> np.ndarray:
        """Вычисляет общую матрицу трансформации"""
        R = np.eye(3, dtype=np.float32)
        
        if self.rotation_y != 0:
            R = rotation_matrix_y(self.rotation_y) @ R
        if self.rotation_x != 0:
            R = rotation_matrix_x(self.rotation_x) @ R
        if self.flip_horizontal:
            R = flip_matrix_x() @ R
        
        return R
    
    def __repr__(self):
        return (f"VirtualCamera(name='{self.name}', "
                f"rot_y={self.rotation_y}°, rot_x={self.rotation_x}°, "
                f"flip={self.flip_horizontal})")


# =============================================
# ПРЕДУСТАНОВКИ (ТОЛЬКО ВИРТУАЛЬНЫЕ УГЛЫ)
# =============================================

PRESET_CAMERAS = {
    # ❌ УБРАЛИ ntu_cam1, ntu_cam2, ntu_cam3 - они уже в файлах!
    'default':  VirtualCamera('default', 0, 0, False),
    # Виртуальные камеры по кругу (дополнительные углы)
    'cam_30':  VirtualCamera('30° Right', 30, 0, False),
    'cam_60':  VirtualCamera('60° Right', 60, 0, False),
    'cam_90':  VirtualCamera('90° Right', 90, 0, False),
    'cam_120': VirtualCamera('120° Right', 120, 0, False),
    'cam_150': VirtualCamera('150° Right', 150, 0, False),
    'cam_180': VirtualCamera('180° Back', 180, 0, False),
    'cam_210': VirtualCamera('210° Left', 210, 0, False),
    'cam_240': VirtualCamera('240° Left', 240, 0, False),
    'cam_270': VirtualCamera('270° Left', 270, 0, False),
    'cam_300': VirtualCamera('300° Left', 300, 0, False),
    'cam_330': VirtualCamera('330° Left', 330, 0, False),
    
    # CCTV-стиль (наклон вниз)
    'cctv_front_15': VirtualCamera('CCTV Front (-15°)', 0, -15, False),
    'cctv_right_90_30':  VirtualCamera('CCTV 90 x, 30y', 90, -30, False),
    'cctv_front_30': VirtualCamera('CCTV Front (-30°)', 0, -30, False),
    'cctv_side_15':  VirtualCamera('CCTV Side (-15°)', 90, -15, False),
    'cctv_side_30':  VirtualCamera('CCTV Side (-30°)', 90, -30, False),
    'cctv_back':     VirtualCamera('CCTV Back', 180, -20, False),
    
    # Отражения
    'mirror_horizontal': VirtualCamera('Mirror Flip', 0, 0, True),
}


# =============================================
# АУГМЕНТАЦИЯ
# =============================================

def augment_skeleton_files(skeleton_files: list,
                          cameras: list,
                          output_format='list') -> dict:
    """
    Аугментирует список скелетных файлов
    
    Args:
        skeleton_files: list of [T, V, 3] numpy arrays
        cameras: list of VirtualCamera
        output_format: 'list' или 'array'
    
    Returns:
        dict с ключами:
            'data': аугментированные скелеты
            'original_indices': индексы исходных файлов
            'camera_names': названия камер
    """
    augmented = []
    original_indices = []
    camera_names = []
    
    print(f"Аугментация {len(skeleton_files)} файлов × {len(cameras)} камер")
    
    for i, skeleton in enumerate(tqdm(skeleton_files, desc="Augmenting")):
        for cam in cameras:
            R = cam.get_transform_matrix()
            aug_skeleton = apply_transform_to_skeleton(skeleton, R, center_joint=0)
            
            augmented.append(aug_skeleton)
            original_indices.append(i)
            camera_names.append(cam.name)
    
    if output_format == 'array':
        # Приводим к единой длине (pad/crop)
        max_T = max(s.shape[0] for s in augmented)
        V = augmented[0].shape[1]
        
        data_array = np.zeros((len(augmented), max_T, V, 3), dtype=np.float32)
        for i, seq in enumerate(augmented):
            T = seq.shape[0]
            data_array[i, :T] = seq
        
        augmented = data_array
    
    return {
        'data': augmented,
        'original_indices': np.array(original_indices),
        'camera_names': camera_names
    }


# =============================================
# ВИЗУАЛИЗАЦИЯ
# =============================================

NTU_SKELETON_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]


def plot_skeleton_3d(skeleton: np.ndarray, 
                    ax: plt.Axes = None,
                    title: str = "",
                    color: str = 'blue') -> plt.Axes:
    """Отрисовывает один кадр скелета в 3D"""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    V = skeleton.shape[0]
    
    # Рисуем суставы
    ax.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], 
              c=color, s=50, alpha=0.8)
    
    # Рисуем кости
    for bone in NTU_SKELETON_BONES:
        if bone[0] < V and bone[1] < V:
            pts = skeleton[bone, :]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                   c=color, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Равные пропорции
    max_range = np.array([
        skeleton[:, 0].max() - skeleton[:, 0].min(),
        skeleton[:, 1].max() - skeleton[:, 1].min(),
        skeleton[:, 2].max() - skeleton[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (skeleton[:, 0].max() + skeleton[:, 0].min()) * 0.5
    mid_y = (skeleton[:, 1].max() + skeleton[:, 1].min()) * 0.5
    mid_z = (skeleton[:, 2].max() + skeleton[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return ax


def visualize_augmentations(skeleton: np.ndarray,
                           cameras: list,
                           frame_idx: int = 0,
                           save_path: str = None):
    """Визуализирует один скелет с разных виртуальных камер"""
    num_cams = len(cameras)
    
    cols = min(4, num_cams)
    rows = (num_cams + cols - 1) // cols
    
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    
    for i, cam in enumerate(cameras):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        R = cam.get_transform_matrix()
        skeleton_aug = apply_transform_to_skeleton(
            skeleton[frame_idx:frame_idx+1], R, center_joint=0
        )
        
        plot_skeleton_3d(skeleton_aug[0], ax, 
                        title=cam.name, 
                        color='royalblue')
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Сохранено: {save_path}")
    
    plt.show()


# =============================================
# 2D ПРОЕКЦИЯ И ВИЗУАЛИЗАЦИЯ
# =============================================

def project_skeleton_2d(skeleton_3d: np.ndarray,
                        view: str = 'front') -> np.ndarray:
    """
    Проецирует 3D скелет [V, 3] в 2D [V, 2] для разных видов.
    
    view:
      - 'front' : смотрим спереди, используем (x, y)
      - 'side'  : смотрим сбоку, используем (z, y)
      - 'top'   : вид сверху, используем (x, z)
    """
    assert skeleton_3d.ndim == 2 and skeleton_3d.shape[1] == 3, "Ожидается [V, 3]"
    x = skeleton_3d[:, 0]
    y = skeleton_3d[:, 1]
    z = skeleton_3d[:, 2]
    
    if view == 'front':
        # Ось X по горизонтали, Y по вертикали
        u, v = x, y
    elif view == 'side':
        # Ось Z по горизонтали, Y по вертикали
        u, v = z, y
    elif view == 'top':
        # Ось X по горизонтали, Z по вертикали (вид сверху)
        u, v = x, z
    else:
        raise ValueError(f"Неизвестный view: {view}")
    
    pts_2d = np.stack([u, v], axis=-1)  # [V, 2]
    return pts_2d


def plot_skeleton_2d(skeleton_2d: np.ndarray,
                     view: str = 'front',
                     ax: plt.Axes = None,
                     title: str = "",
                     color: str = 'blue'):
    """
    Рисует 2D скелет (суставы и кости) на плоскости.
    
    skeleton_2d: [V, 2]
    view: 'front' | 'side' | 'top' (только для подписи)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    V = skeleton_2d.shape[0]
    
    # Рисуем суставы
    ax.scatter(skeleton_2d[:, 0], skeleton_2d[:, 1], 
               c=color, s=40, alpha=0.9, zorder=2)
    
    # Рисуем кости
    for bone in NTU_SKELETON_BONES:
        if bone[0] < V and bone[1] < V:
            pts = skeleton_2d[list(bone), :]
            ax.plot(pts[:, 0], pts[:, 1], 
                    c=color, linewidth=2, alpha=0.7, zorder=1)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{title} ({view})")
    
    # Инвертировать ось Y, если хочешь, как в изображениях (0 сверху)
    # ax.invert_yaxis()
    
    ax.grid(True, alpha=0.2)
    return ax


def visualize_2d_views_for_cameras(skeleton_seq: np.ndarray,
                                   cameras: list,
                                   frame_idx: int = 0,
                                   base_view: str = 'front',
                                   save_path: str = None):
    """
    Показывает 2D проекцию одного кадра:
      - сначала оригинал (без виртуальной камеры)
      - затем тот же кадр после трансформаций камер
    
    Args:
        skeleton_seq: [T, V, 3] - последовательность скелета
        cameras: список VirtualCamera
        frame_idx: какой кадр брать
        base_view: 'front' | 'side' | 'top' - на какую плоскость проецируем
        save_path: путь для сохранения картинки (png)
    """
    T, V, C = skeleton_seq.shape
    assert C == 3
    
    frame = skeleton_seq[frame_idx]  # [V, 3]
    
    num_cams = len(cameras)
    cols = min(3, num_cams + 1)  # +1 для оригинала
    rows = (num_cams + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    
    axes = axes.reshape(rows, cols)
    
    # 1. Оригинальная проекция
    ax0 = axes[0, 0]
    orig_2d = project_skeleton_2d(frame, view=base_view)
    plot_skeleton_2d(orig_2d, view=base_view, ax=ax0, 
                     title="Original", color='green')
    
    # 2. Виртуальные камеры
    idx = 1
    for cam in cameras:
        r = idx // cols
        c = idx % cols
        
        if r >= rows:
            break
        
        ax = axes[r, c]
        
        R = cam.get_transform_matrix()
        frame_aug = apply_transform_to_skeleton(
            skeleton_seq[frame_idx:frame_idx+1], R, center_joint=0
        )[0]  # [V, 3]
        
        frame_aug_2d = project_skeleton_2d(frame_aug, view=base_view)
        plot_skeleton_2d(frame_aug_2d, view=base_view, ax=ax,
                         title=cam.name, color='royalblue')
        
        idx += 1
    
    # Удаляем пустые оси, если есть
    for k in range(idx, rows * cols):
        r = k // cols
        c = k % cols
        fig.delaxes(axes[r, c])
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 2D визуализация сохранена: {save_path}")
    
    plt.show()


# =============================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# =============================================

if __name__ == '__main__':
    print("="*70)
    print("NTU RGB+D Skeleton Data Augmentation")
    print("="*70)
    
    # ========== ЧТЕНИЕ .SKELETON ФАЙЛОВ ==========
    
    DATASET_PATH = 'ntu_cough_dataset'
    
    # Читаем все файлы из test/cough
    print("\n📂 Загрузка test/cough...")
    try:
        test_cough_data, test_cough_files = load_skeleton_dataset(
            DATASET_PATH, 
            split='test', 
            class_name='cough'
        )
        
        print(f"\n✓ Загружено {len(test_cough_data)} файлов")
        print(f"Пример: {test_cough_files[0].name}")
        
        # Парсим информацию из имени файла
        info = parse_ntu_filename(test_cough_files[0])
        print(f"  Setup: {info['setup']}")
        print(f"  Camera: {info['camera']} (реальная камера NTU)")
        print(f"  Action: {info['action']}")
        
        # Статистика по длинам
        lengths = [s.shape[0] for s in test_cough_data]
        print(f"\nСтатистика длин:")
        print(f"  Мин: {min(lengths)} кадров")
        print(f"  Макс: {max(lengths)} кадров")
        print(f"  Среднее: {np.mean(lengths):.1f} кадров")
        
    except Exception as e:
        print(f"⚠️  Ошибка: {e}")
        print("\nПроверьте путь к датасету!")
        exit(1)
    
    print(f"Пример: {test_cough_files[7].name}")
    example_skeleton = test_cough_data[7]   # [T, V, 3]
    mid_frame = example_skeleton.shape[0] // 2

    # Выбираем виртуальные камеры
    selected_cameras = [
        # PRESET_CAMERAS['default'],
        PRESET_CAMERAS['cctv_front_15'],
        PRESET_CAMERAS['cctv_right_90_30'],
        PRESET_CAMERAS['cctv_front_30'],
        PRESET_CAMERAS['cctv_side_30'],
        PRESET_CAMERAS['cctv_back'],
    ]

    # 2D визуализация на фронтальной плоскости (x,y)
    visualize_2d_views_for_cameras(
        example_skeleton,
        selected_cameras,
        frame_idx=mid_frame,
        base_view='front',          # можно 'side' или 'top'
        save_path='2d_views_front.png'
    )

    
    # ========== НАСТРОЙКА ВИРТУАЛЬНЫХ КАМЕР ==========
    if False:
        print("\n" + "="*70)
        print("ВИРТУАЛЬНЫЕ КАМЕРЫ")
        print("="*70)
        
        # Выбираем камеры (БЕЗ дублирования реальных углов NTU)
        selected_cameras = [
            PRESET_CAMERAS['default'],
            PRESET_CAMERAS['cam_90'],   # Правый бок (90°)
            PRESET_CAMERAS['cam_180'],  # Сзади (180°)
            PRESET_CAMERAS['cam_270'],  # Левый бок (270°)
            PRESET_CAMERAS['cctv_front_15'],  # CCTV спереди
            PRESET_CAMERAS['mirror_horizontal'],  # Зеркало
        ]
        
        print(f"\nВыбрано {len(selected_cameras)} виртуальных камер:")
        for cam in selected_cameras:
            print(f"  • {cam}")
        
        print(f"\nВажно: реальные NTU камеры (C001=-45°, C002=0°, C003=+45°)")
        print(f"        уже записаны в отдельных .skeleton файлах!")
    
    
    # ========== ВИЗУАЛИЗАЦИЯ ==========
    if False:
        print("\n" + "="*70)
        print("ВИЗУАЛИЗАЦИЯ")
        print("="*70)
        
        # Берём первый файл
        example_skeleton = test_cough_data[4]
        mid_frame = example_skeleton.shape[0] // 2
        
        print(f"\nПример: {test_cough_files[0].name}")
        print(f"Кадров: {example_skeleton.shape[0]}, показываем кадр {mid_frame}")
        
        visualize_augmentations(
            example_skeleton,
            selected_cameras,
            frame_idx=mid_frame,
            save_path='augmentation_test_cough.png'
        )
        
        # ========== ПОЛНАЯ АУГМЕНТАЦИЯ ==========
        
        print("\n" + "="*70)
        print("АУГМЕНТАЦИЯ")
        print("="*70)
        
        response = input("\nАугментировать все файлы? (y/n): ")
        
        if response.lower() == 'y':
            result = augment_skeleton_files(
                test_cough_data,
                selected_cameras,
                output_format='list'
            )
            
            print(f"\n✓ Готово!")
            print(f"Исходных файлов: {len(test_cough_data)}")
            print(f"После аугментации: {len(result['data'])}")
            print(f"Коэффициент: {len(result['data']) / len(test_cough_data):.1f}x")
            
            # Сохранение
            output_dir = Path('augmented_test_cough')
            output_dir.mkdir(exist_ok=True)
            
            # Сохраняем как pickle (проще для списков разной длины)
            import pickle
            with open(output_dir / 'augmented_data.pkl', 'wb') as f:
                pickle.dump(result, f)
            
            print(f"✓ Сохранено в {output_dir}/augmented_data.pkl")
