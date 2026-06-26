"""Character set for LamSonOCR - Japanese + English + Numbers."""


class Charset:
    """Maps characters to indices for CTC-based OCR."""

    BLANK = "<BLANK>"

    # Define character groups
    DIGITS = "0123456789"
    ENGLISH_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ENGLISH_LOWER = "abcdefghijklmnopqrstuvwxyz"
    HIRAGANA = (
        "あいうえおかきくけこさしすせそたちつてとなにぬねの"
        "はひふへほまみむめもやゆよらりるれろわをん"
        "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
        "ぁぃぅぇぉっゃゅょ"
    )
    KATAKANA = (
        "アイウエオカキクケコサシスセソタチツテトナニヌネノ"
        "ハヒフヘホマミムメモヤユヨラリルレロワヲン"
        "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ"
        "ァィゥェォッャュョヴー"
    )
    KANJI_COMMON = (
        "年月日人車台号番出品初度登録定員内装外色型式"
        "車名走行距離評価点検査有効期間排気量燃料"
        "自動手動無段変速機前後左右上下新中古"
        "令和平成昭和大正"
        "万千百十一二三四五六七八九零"
        "東西南北海道府県市区町村"
        "通常特別限定"
    )
    SYMBOLS = " /-()[]・。、:;\"'.!?@#%&*+=~^_|\\<>{}"

    def __init__(self):
        # Build charset: BLANK at index 0
        chars = []
        for group in [
            self.DIGITS,
            self.ENGLISH_UPPER,
            self.ENGLISH_LOWER,
            self.HIRAGANA,
            self.KATAKANA,
            self.KANJI_COMMON,
            self.SYMBOLS,
        ]:
            for c in group:
                if c not in chars:
                    chars.append(c)

        self._idx_to_char = {0: self.BLANK}
        self._char_to_idx = {self.BLANK: 0}
        for i, c in enumerate(chars, start=1):
            self._idx_to_char[i] = c
            self._char_to_idx[c] = i

    @property
    def blank_idx(self) -> int:
        return 0

    @property
    def num_classes(self) -> int:
        return len(self._idx_to_char)

    def encode(self, text: str) -> list[int]:
        """Encode text string to list of character indices."""
        result = []
        for c in text:
            if c in self._char_to_idx:
                result.append(self._char_to_idx[c])
            # Skip unknown characters
        return result

    def decode(self, indices: list[int]) -> str:
        """Decode list of indices to text string (no CTC logic)."""
        return "".join(
            self._idx_to_char.get(i, "") for i in indices if i != self.blank_idx
        )

    def ctc_decode(self, indices: list[int]) -> str:
        """Decode CTC output: remove blanks and collapse repeats."""
        result = []
        prev = None
        for idx in indices:
            if idx == self.blank_idx:
                prev = None
                continue
            if idx != prev:
                result.append(idx)
            prev = idx
        return self.decode(result)
