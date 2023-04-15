__all__ = ['progress_bar']

def progress_bar(current: float, total: float, total_steps: int = 50) -> str:
    current_pct = current / total
    current_steps = round(total_steps * current_pct)

    bar = ''

    for _ in range(current_steps - 1):
        bar += '='
    
    bar += '>'

    for _ in range(total_steps - current_steps):
        bar += '.'
    
    return bar

if __name__ == '__main__':
    print(progress_bar(25, 100))

