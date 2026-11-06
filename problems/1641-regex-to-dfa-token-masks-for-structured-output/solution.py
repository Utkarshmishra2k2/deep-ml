from collections import deque

def compile_regex(pattern):
    """Compile the supported regex dialect into a deterministically numbered DFA."""

    def split_alternatives(text):
        alternatives = []
        start = 0
        i = 0
        in_class = False

        while i < len(text):
            ch = text[i]
            if ch == "\\":
                i += 2  # The escaped character cannot be a separator.
                continue
            if ch == "[":
                in_class = True
            elif ch == "]":
                in_class = False
            elif ch == "|" and not in_class:
                alternatives.append(text[start:i])
                start = i + 1
            i += 1

        if i > len(text) and text.endswith("\\"):
            raise ValueError("Pattern ends with an incomplete escape")

        alternatives.append(text[start:])
        return alternatives

    def parse_class(text, i):
        # Parse a class starting at text[i] and return (characters, next_index).
        i += 1
        items = []

        while i < len(text) and text[i] != "]":
            if text[i] == "\\":
                i += 1
                if i == len(text):
                    raise ValueError("Incomplete escape in character class")
                items.append((text[i], False))
            else:
                items.append((text[i], text[i] == "-"))
            i += 1

        if i == len(text):
            raise ValueError("Unclosed character class")
        if not items:
            raise ValueError("Empty character class")

        chars = set()
        j = 0
        while j < len(items):
            if (
                j + 2 < len(items)
                and items[j + 1][1]
                and not items[j][1]
                and not items[j + 2][1]
            ):
                lo, hi = ord(items[j][0]), ord(items[j + 2][0])
                if lo > hi:
                    raise ValueError("Descending character range")
                chars.update(chr(code) for code in range(lo, hi + 1))
                j += 3
            else:
                chars.add(items[j][0])
                j += 1

        return chars, i + 1

    def parse_atoms(alternative):
        atoms = []
        i = 0

        while i < len(alternative):
            ch = alternative[i]

            if ch == "\\":
                i += 1
                if i == len(alternative):
                    raise ValueError("Pattern ends with an incomplete escape")
                atom_chars = {alternative[i]}
                i += 1
            elif ch == "[":
                atom_chars, i = parse_class(alternative, i)
            elif ch in "*+?":
                raise ValueError("Quantifier must follow an atom")
            else:
                atom_chars = {ch}
                i += 1

            quantifier = ""
            if i < len(alternative) and alternative[i] in "*+?":
                quantifier = alternative[i]
                i += 1

            atoms.append((atom_chars, quantifier))

        return atoms

    alternatives = [parse_atoms(a) for a in split_alternatives(pattern)]

    # Build an NFA with epsilon transitions.
    epsilon = [set()]
    nfa_trans = [{}]

    def new_state():
        epsilon.append(set())
        nfa_trans.append({})
        return len(epsilon) - 1

    def add_char_edge(src, ch, dst):
        nfa_trans[src].setdefault(ch, set()).add(dst)

    nfa_start = 0
    nfa_accept = None

    for atoms in alternatives:
        current = new_state()
        epsilon[nfa_start].add(current)

        for atom_chars, quantifier in atoms:
            nxt = new_state()

            for ch in atom_chars:
                add_char_edge(current, ch, nxt)

            if quantifier in ("*", "+"):
                for ch in atom_chars:
                    add_char_edge(nxt, ch, nxt)

            if quantifier in ("*", "?"):
                epsilon[current].add(nxt)

            current = nxt

        if nfa_accept is None:
            nfa_accept = new_state()
        epsilon[current].add(nfa_accept)

    alphabet = sorted({ch for state_edges in nfa_trans for ch in state_edges})

    def epsilon_closure(states):
        closure = set(states)
        stack = list(states)

        while stack:
            state = stack.pop()
            for nxt in epsilon[state]:
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)

        return frozenset(closure)

    start_subset = epsilon_closure({nfa_start})
    subset_to_id = {start_subset: 0}
    queue = deque([start_subset])
    dfa_trans = {}
    accepting = []

    while queue:
        subset = queue.popleft()
        state_id = subset_to_id[subset]

        if nfa_accept in subset:
            accepting.append(state_id)

        for ch in alphabet:
            moved = set()
            for nfa_state in subset:
                moved.update(nfa_trans[nfa_state].get(ch, ()))

            if not moved:
                continue

            next_subset = epsilon_closure(moved)
            if next_subset not in subset_to_id:
                subset_to_id[next_subset] = len(subset_to_id)
                queue.append(next_subset)

            dfa_trans[(state_id, ch)] = subset_to_id[next_subset]

    return {
        "n_states": len(subset_to_id),
        "start": 0,
        "accepting": sorted(accepting),
        "trans": dfa_trans,
    }

def advance(dfa, state, token):
    """Return the state after consuming token, or None if it is rejected."""
    if not isinstance(state, int) or not 0 <= state < dfa["n_states"]:
        return None

    for ch in token:
        state = dfa["trans"].get((state, ch))
        if state is None:
            return None

    return state

def token_masks(dfa, vocab):
    """Return a boolean token-validity mask for each DFA state."""
    return [
        [advance(dfa, state, token) is not None for token in vocab]
        for state in range(dfa["n_states"])
    ]
