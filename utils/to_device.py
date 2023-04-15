from torch import device as Device

__all__ = ['to_device']

def to_device(data, device: Device) :
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    
    return data.to(device, non_blocking=True)