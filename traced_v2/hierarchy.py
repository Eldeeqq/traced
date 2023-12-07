from networkx import DiGraph

from traced_v2.utils import create_hash, remove_duplicates


class HashHierarchy:
    def __init__(self, max_size: int | None = None):
        self.hash_graph = DiGraph()
        self.hash_to_path = {}
        self.max_size: int | None = None

    def hash(self, path: list[str], max_size: int | None = None):
        if not isinstance(path[0], str):
            path = [str(p) for p in path]

        max_size = max_size or self.max_size or 8

        path = remove_duplicates(path)
        path_hash = create_hash("".join(path), max_size)

        if path_hash in self.hash_to_path:
            self.hash_graph.nodes[path_hash]["count"] += 1
            return path_hash

        self.hash_to_path[path_hash] = path

        path_copy = list(path)
        hash = str(path_hash)

        while path_copy:
            path_copy = path_copy[:-1]
            curr_hash = create_hash("".join(path_copy), max_size)
            self.hash_graph.add_edge(curr_hash, hash)
            hash = curr_hash

            if not path_copy:
                break

        self.hash_graph.nodes[path_hash]["count"] = 1
        return path_hash
