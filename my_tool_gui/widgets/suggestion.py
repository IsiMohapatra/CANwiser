class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.original_words = set()

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.all_frames = set()          # All raw frames (for fallback)
        self.root_frames = set()         # Only root-level signal names
        self.full_to_root_map = {}       # Map full names to root signals

    def extract_root_signal(self, signal_name):
        """
        Removes 'DINH' signals and anything after the second underscore.
        For example:
            'DINH_stFId.FId_bInh_Sig435h_Com_uHvbMaxCell_432_ini' → ''
            'SomePrefix_Signal_Extra_Info' → 'SomePrefix_Signal'
        """
        if signal_name.startswith("DINH"):
            return ""

        parts = signal_name.split("_")
        if len(parts) >= 2:
            return "_".join(parts[:2])  # Keep only first two parts
        return signal_name

    def insert(self, word):
        """
        Insert full word but store root signal as primary match.
        Skips 'DINH' and invalid root names.
        """
        self.all_frames.add(word)

        root_word = self.extract_root_signal(word)
        if not root_word:
            return  # Skip inserting 'DINH' or malformed signals

        self.root_frames.add(root_word)
        self.full_to_root_map[word] = root_word

        lower_root = root_word.lower()
        node = self.root

        for char in lower_root:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end_of_word = True
        node.original_words.add(root_word)

    def search(self, text):
        """
        Search for signals that start with or contain the search text.
        Always returns root signal names only.
        """
        text = text.lower().strip()
        if not text:
            return list(self.root_frames)

        matching_frames = set()

        node = self.root
        for char in text:
            if char not in node.children:
                return self.search_substring(text)
            node = node.children[char]

        matching_frames.update(self._collect_original_words(node))
        matching_frames.update(self.search_substring(text))

        return list(matching_frames)

    def search_substring(self, text):
        """
        Fallback search — substring match in root signal names.
        """
        return [signal for signal in self.root_frames if text in signal.lower()]

    def _collect_original_words(self, node):
        """
        Recursively collect all original root signal names from a node.
        """
        words = list(node.original_words) if node.is_end_of_word else []
        for child_node in node.children.values():
            words.extend(self._collect_original_words(child_node))
        return list(set(words))

    def get_all_frames(self):
        """
        Get all full signals.
        """
        return list(self.all_frames)

    def get_all_root_frames(self):
        """
        Get all deduplicated root signals.
        """
        return list(self.root_frames)