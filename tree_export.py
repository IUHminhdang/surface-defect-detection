import os

def print_tree(start_path='.', indent=''):
    items = sorted(os.listdir(start_path))
    # Lọc bỏ các thư mục hệ thống hoặc không mong muốn nếu cần (ví dụ: .git, __pycache__, node_modules)
    ignore = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
    items = [item for item in items if item not in ignore]
    
    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        is_last = (i == len(items) - 1)
        connector = '└── ' if is_last else '├── '
        print(indent + connector + item)
        
        if os.path.isdir(path):
            extension = '    ' if is_last else '│   '
            print_tree(path, indent + extension)

if __name__ == '__main__':
    print("Cấu trúc thư mục:")
    print_tree('.')
