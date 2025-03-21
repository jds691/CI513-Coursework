from os import system, name

def clear_console() -> None:
    # Windows
    if name == 'nt':
        _ = system('cls')

    # Unix
    else:
        _ = system('clear')