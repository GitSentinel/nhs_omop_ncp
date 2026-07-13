import torch
print(f'CUDA available : {torch.cuda.is_available()}')
print(f'Device count   : {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
