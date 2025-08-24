# 단어 단위
noun_indicators = [
    # ------------- English -------------
    'could be',
    'so',
    'is',
    'thus',
    'therefore',
    'final',
    'answer',
    'result',
    # ------------- Korean -------------
    '그래서',
    '따라서',
    '그러므로',
    '최종',
    '답',
    '결과',
    '정답',
    '은',
    '는',
    '답은',
    '정답은'
]
# 한국어 종결어미 단위
sentence_end_suffixes = [
    '이다',
    '일 수 있다'
]
TRIVIAL_PATTERNS = [":", ",", ".", "!", "?", ";", ":", "'"]
INDICATORS_OF_KEYS = [f"{indicator} " for indicator in noun_indicators] + sentence_end_suffixes

__all__ = [
    "INDICATORS_OF_KEYS",
    "TRIVIAL_PATTERNS"
]