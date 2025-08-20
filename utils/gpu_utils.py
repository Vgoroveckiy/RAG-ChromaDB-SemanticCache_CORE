import torch


def gpu_available(min_memory_mb: int = 500) -> bool:
    """
    Проверяет доступность GPU с учетом:
    - Наличия CUDA
    - Совместимости архитектуры (sm_70+)
    - Достаточного объема свободной памяти
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Проверка совместимости архитектуры
        capability = torch.cuda.get_device_capability(0)
        if capability[0] < 7:  # Требуется минимум sm_70
            return False

        # Проверка свободной памяти
        free_mem = torch.cuda.mem_get_info(0)[0] / (1024**2)  # MB
        return free_mem >= min_memory_mb
    except Exception as e:
        print(f"GPU check error: {str(e)}")
        return False
