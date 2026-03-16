# Ordinal Encoding Function
def ordinal_encode(data):
    unique = []

    # Find unique categories
    for item in data:
        if item not in unique:
            unique.append(item)

    # Create mapping for categories
    mapping = {value: i for i, value in enumerate(unique)}

    # Encode the data
    encoded = [mapping[item] for item in data]

    return mapping, encoded


# One-Hot Encoding Function
def one_hot_encode(data):
    unique = []

    # Find unique categories
    for item in data:
        if item not in unique:
            unique.append(item)

    encoded = []

    # Create one-hot vectors
    for item in data:
        row = []
        for category in unique:
            row.append(1 if item == category else 0)
        encoded.append(row)

    return unique, encoded


# Example dataset
data = ["Red", "Blue", "Green", "Blue","yellow","red","blue","green","yellow"]


# Calling functions
print("Ordinal Encoding:", ordinal_encode(data))
print("One Hot Encoding:", one_hot_encode(data))