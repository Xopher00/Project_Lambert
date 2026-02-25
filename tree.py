class Tree:
    """Structural recursion for nested parameter trees."""

    class Pair:
        """Wrapper for (param, grad) that stops recursion."""
        __slots__ = ("fst", "snd")

        def __init__(self, a, b):
            self.fst = a
            self.snd = b

        def __repr__(self):
            return f"Tree.Pair({self.fst}, {self.snd})"

    # === TreeZip ===
    @staticmethod
    def zip(a, b):
        if isinstance(a, dict):
            return {k: Tree.zip(a[k], b[k]) for k in a}
        if isinstance(a, tuple):
            return tuple(Tree.zip(x, y) for x, y in zip(a, b))
        if isinstance(a, list):
            return [Tree.zip(x, y) for x, y in zip(a, b)]
        return Tree.Pair(a, b)

    # === TreeMap ===
    @staticmethod
    def map(fn, x):
        if isinstance(x, Tree.Pair):
            return fn(x)
        if isinstance(x, dict):
            return {k: Tree.map(fn, v) for k, v in x.items()}
        if isinstance(x, tuple):
            return tuple(Tree.map(fn, v) for v in x)
        if isinstance(x, list):
            return [Tree.map(fn, v) for v in x]
        return fn(x)

    # === TreeFold ===
    @staticmethod
    def fold(fn, tree, default=None):
        if tree is None:
            return default
        if isinstance(tree, Tree.Pair):
            fst_folded = Tree.fold(fn, tree.fst, default)
            snd_folded = Tree.fold(fn, tree.snd, default)
            return fn('pair', fst_folded, snd_folded)
        if isinstance(tree, dict):
            folded = {k: Tree.fold(fn, v, default) for k, v in tree.items()}
            return fn('dict', folded)
        if isinstance(tree, list):
            folded = [Tree.fold(fn, item, default) for item in tree]
            return fn('list', folded)
        if isinstance(tree, tuple):
            folded = tuple(Tree.fold(fn, item, default) for item in tree)
            return fn('tuple', folded)
        return fn('leaf', tree)

    # === Tree Metrics ===
    @staticmethod
    def depth(tree):
        """Maximum depth of tree structure"""
        def folder(tag, *args):
            if tag == 'pair':
                return 1 + max(args[0], args[1])
            elif tag == 'list':
                return 1 + max(args[0]) if args[0] else 0
            elif tag == 'dict':
                values = list(args[0].values())
                return 1 + max(values) if values else 0
            return 0
        return Tree.fold(folder, tree, 0)

    @staticmethod
    def size(tree):
        """Count all nodes in tree (Pairs, dict branches, list items)"""
        def folder(tag, *args):
            if tag == 'pair':
                return 1 + args[0] + args[1]
            elif tag == 'list':
                return sum(args[0]) if args[0] else 0
            elif tag == 'dict':
                values = list(args[0].values())
                return 1 + sum(values) if values else 0
            return 0
        return Tree.fold(folder, tree, 0)
