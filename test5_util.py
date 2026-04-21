def count_str(txt):
    return len(txt.split())

def read_file(fname):
    with open(fname,"r",encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    print("main program")