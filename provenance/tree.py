class Tree:
    """
    Structural recursion over nested proof trees.

    Provides zip, map, and fold over trees built from Pair nodes, dicts,
    lists, and tuples. Used by Provenance to construct and traverse proof
    trees, and by Audit to extract and format reasoning paths.
    """

    class Pair:
        """
        A binary node holding two branches of a proof tree.

        Used to represent a single reasoning step: fst is the left
        sub-proof and snd is the right sub-proof. Leaf values are
        integers (entity indices).
        """
        __slots__ = ("fst", "snd")

        def __init__(self, a, b):
            self.fst = a
            self.snd = b

        def __repr__(self):
            return f"Tree.Pair({self.fst}, {self.snd})"

    # === TreeZip ===
    @staticmethod
    def zip(a, b):
        """
        Pair corresponding leaves of two identically shaped trees.

        Recursively traverses both trees in lockstep, wrapping each pair
        of leaves in a Tree.Pair. The two trees must have the same structure.
        """
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
        """
        Apply a function to every leaf in the tree.

        Recursively traverses the tree, applying fn to each Tree.Pair or
        scalar leaf. Dicts, tuples, and lists are traversed but not
        transformed themselves.
        """
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
        """
        Reduce a proof tree to a single value.

        Recursively folds the tree bottom-up, calling fn at each node
        with a tag and the already-folded children. Tags are:
        ``'pair'``, ``'dict'``, ``'list'``, ``'tuple'``, ``'leaf'``.
        Returns default for None nodes.
        """
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
