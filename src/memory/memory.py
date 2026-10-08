import json
import os


MEMORY_FILE = os.path.join(
    os.path.dirname(__file__),
    "memory.json"
)


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}

    with open(MEMORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_memory(key, value):
    memory = load_memory()

    memory[key] = value

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(memory, file, indent=4)

    return {
        "success": True,
        "message": f"Memory saved: {key}"
    }


def get_memory(key):
    memory = load_memory()

    return {
        "success": True,
        "key": key,
        "value": memory.get(key)
    }


def get_all_memories():
    return load_memory()


if __name__ == "__main__":

    print(save_memory(
        "preferred_language",
        "Hindi"
    ))

    print(get_memory(
        "preferred_language"
    ))

    print(get_all_memories())