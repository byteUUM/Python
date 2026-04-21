from test5_util import count_str, read_file
from pathlib import Path

def main():
    p = Path("test5_prompt.txt")
    if(p.exists()):
        str = read_file(p)
        num = count_str(str)
        print(num)
    else:
        print("test5_prompt.txt not exists")

if __name__ == "__main__":
    main()