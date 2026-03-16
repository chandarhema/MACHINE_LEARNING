"""Implement ordinal encoding and one-hot encoding methods in Python from scratch.
"""


def ordinal_encoding(data):
    unique = []
    for item in data:
        if item not in unique:
            unique.append(item)
    print(unique)
    mapping = {value: i for i, value in enumerate(unique)}
    encoded = [mapping[item] for item in data]
    print(encoded)
    print(mapping)

def onehot_encoding(data):
    

def main():
    data = ["red","blue","green","white","black","pink","purple","yellow","pink","blue","green","white","black","pink"]
    ordinal_encoding(data)



if __name__ == "__main__":
    main()


