import pickle


def save_object(file_path, obj):

    with open(file_path, "wb") as file:

        pickle.dump(obj, file)


def load_object(file_path):

    with open(file_path, "rb") as file:

        loaded_object = pickle.load(file)

    return loaded_object