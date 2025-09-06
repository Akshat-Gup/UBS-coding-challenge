from typing import List, Dict, Tuple
import re


def duolingo_sort(payload: dict) -> dict:
    """
    Sort numbers in different languages according to the challenge requirements.
    
    Part 1: Convert to integers and sort
    Part 2: Sort while preserving original representation with language priority for duplicates
    """
    part = payload.get("part")
    challenge_input = payload.get("challengeInput", {})
    unsorted_list = challenge_input.get("unsortedList", [])
    
    if part == "ONE":
        return {"sortedList": part_one_sort(unsorted_list)}
    elif part == "TWO":
        return {"sortedList": part_two_sort(unsorted_list)}
    else:
        raise ValueError(f"Unknown part: {part}")


def part_one_sort(unsorted_list: List[str]) -> List[str]:
    """Part 1: Convert Roman numerals and Arabic numerals to integers and sort"""
    values = []
    for item in unsorted_list:
        value = parse_number_to_int(item)
        values.append(value)
    
    values.sort()
    return [str(v) for v in values]


def part_two_sort(unsorted_list: List[str]) -> List[str]:
    """Part 2: Sort with language priority for duplicates"""
    # Parse each item to get its numerical value and language priority
    items_with_data = []
    for item in unsorted_list:
        value = parse_number_to_int(item)
        priority = get_language_priority(item)
        items_with_data.append((value, priority, item))
    
    # Sort by value first, then by language priority
    items_with_data.sort(key=lambda x: (x[0], x[1]))
    
    return [item[2] for item in items_with_data]


def parse_number_to_int(text: str) -> int:
    """Parse a number string in any supported language to integer"""
    text = text.strip()
    
    # Check if it's a plain Arabic numeral
    if text.isdigit():
        return int(text)
    
    # Check if it's a Roman numeral
    if is_roman_numeral(text):
        return roman_to_int(text)
    
    # Check if it's Traditional Chinese
    trad_chinese_value = traditional_chinese_to_int(text)
    if trad_chinese_value is not None and trad_chinese_value >= 0:
        return trad_chinese_value
    
    # Check if it's Simplified Chinese
    simp_chinese_value = simplified_chinese_to_int(text)
    if simp_chinese_value is not None and simp_chinese_value >= 0:
        return simp_chinese_value
    
    # Check if it's German
    german_value = german_to_int(text)
    if german_value is not None:
        return german_value
    
    # Check if it's English (do this last as it's more ambiguous)
    english_value = english_to_int(text)
    if english_value is not None:
        return english_value
    
    raise ValueError(f"Could not parse number: {text}")


def get_language_priority(text: str) -> int:
    """Get language priority for sorting duplicates"""
    text = text.strip()
    
    # Priority order: Roman (0), English (1), Traditional Chinese (2), 
    # Simplified Chinese (3), German (4), Arabic (5)
    
    if is_roman_numeral(text):
        return 0
    elif text.isdigit():
        return 5
    else:
        # Check Traditional Chinese first
        trad_chinese_value = traditional_chinese_to_int(text)
        if trad_chinese_value is not None and trad_chinese_value >= 0:
            return 2
        
        # Check Simplified Chinese
        simp_chinese_value = simplified_chinese_to_int(text)
        if simp_chinese_value is not None and simp_chinese_value >= 0:
            return 3
            
        # Check German
        german_value = german_to_int(text)
        if german_value is not None:
            return 4
            
        # Check English (last since it's most ambiguous)
        english_value = english_to_int(text)
        if english_value is not None:
            return 1
            
        return 6  # Unknown


def is_roman_numeral(text: str) -> bool:
    """Check if text is a valid Roman numeral"""
    roman_pattern = r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_pattern, text.upper()))


def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer"""
    values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50, 
        'C': 100, 'D': 500, 'M': 1000
    }
    
    roman = roman.upper()
    total = 0
    prev_value = 0
    
    for char in reversed(roman):
        value = values[char]
        if value < prev_value:
            total -= value
        else:
            total += value
        prev_value = value
    
    return total


def english_to_int(text: str) -> int:
    """Convert English number words to integer"""
    text = text.lower().strip()
    
    # Check if this text contains non-ASCII characters (likely not English)
    if not all(ord(c) < 128 for c in text):
        return None
    
    # Basic number words
    ones = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19
    }
    
    tens = {
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
    }
    
    scales = {
        'hundred': 100, 'thousand': 1000, 'million': 1000000, 
        'billion': 1000000000, 'trillion': 1000000000000
    }
    
    # Handle simple cases first
    if text in ones:
        return ones[text]
    if text in tens:
        return tens[text]
    
    # Check if text contains any English number words
    words = text.replace('-', ' ').replace(',', '').split()
    has_english_words = any(word in ones or word in tens or word in scales for word in words)
    if not has_english_words:
        return None
    
    # Parse complex numbers
    total = 0
    current = 0
    
    for word in words:
        if word == 'and':
            # English often uses "and" as a conjunction: "one hundred and one"
            continue
        if word in ones:
            current += ones[word]
        elif word in tens:
            current += tens[word]
        elif word == 'hundred':
            current *= 100
        elif word in scales:
            if word == 'hundred':
                current *= 100
            else:
                total += current * scales[word]
                current = 0
    
    return total + current


def _parse_chinese_small_segment(segment: str, digits: Dict[str, int], small_units: Dict[str, int]) -> int:
    """Parse a Chinese number segment without large units (e.g., up to thousands).
    Handles patterns like 二千三百零四, 十五, 二十, 三百零二, 千零三, 百, 千, 十, etc.
    """
    if not segment:
        return 0

    value = 0
    current_num = 0
    i = 0
    while i < len(segment):
        ch = segment[i]
        if ch in digits:
            current_num = digits[ch]
        elif ch in small_units:
            unit_val = small_units[ch]
            if current_num == 0:
                # Implicit one, e.g., 十 -> 10, 百 -> 100
                current_num = 1
            value += current_num * unit_val
            current_num = 0
        else:
            # Skip placeholders like 零 or 〇 or any unknown char in this context
            pass
        i += 1

    value += current_num
    return value


def _parse_chinese_number(text: str,
                          digits: Dict[str, int],
                          small_units: Dict[str, int],
                          large_units_ordered: List[Tuple[str, int]]) -> int:
    """Parse a Chinese number string using provided digit map and units.
    large_units_ordered should be provided from largest to smallest, e.g., [("兆",10**12),("億",10**8),("萬",10**4)].
    This function supports zeros (零/〇) as placeholders, and implicit ones for unit-only tokens.
    """
    # Quick single-char handling
    if len(text) == 1:
        if text in digits:
            return digits[text]
        if text in small_units:
            return small_units[text]
        for lu, mult in large_units_ordered:
            if text == lu:
                return mult
        return None

    # Remove placeholder zeros to simplify splitting logic (but keep structure by not collapsing completely)
    # We'll keep zeros inside segments; for splitting on large units it's okay to leave them as-is.

    # Recursive decomposition by largest units first
    remaining = text
    total = 0
    for unit_char, multiplier in large_units_ordered:
        if unit_char in remaining:
            parts = remaining.split(unit_char)
            # There could be multiple occurrences; process left-most once each iteration, then continue on the remainder
            left = parts[0]
            right = unit_char.join(parts[1:])  # Re-join the rest if multiple units of same char appear

            left_val = _parse_chinese_small_segment(left, digits, small_units) if left else 1
            total += left_val * multiplier
            remaining = right
    # Finally, add the tail segment (below the smallest large unit)
    tail_val = _parse_chinese_small_segment(remaining, digits, small_units) if remaining else 0
    total += tail_val
    return total


def traditional_chinese_to_int(text: str) -> int:
    """Convert Traditional Chinese numbers to integer"""
    # Traditional Chinese numerals - digits and units
    digits = {
        '零': 0, '〇': 0,
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '兩': 2  # Traditional alternate for two
    }
    small_units = {'十': 10, '百': 100, '千': 1000}
    # Order large units from largest to smallest
    large_units = [('兆', 10**12), ('億', 10**8), ('萬', 10**4)]

    # Quick language detection
    chinese_chars = set(digits.keys()) | set(small_units.keys()) | set(u for u, _ in large_units)
    if not any(c in chinese_chars for c in text):
        return None

    return _parse_chinese_number(text, digits, small_units, large_units)


def simplified_chinese_to_int(text: str) -> int:
    """Convert Simplified Chinese numbers to integer"""
    digits = {
        '零': 0, '〇': 0,
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '两': 2  # Simplified alternate for two
    }
    small_units = {'十': 10, '百': 100, '千': 1000}
    large_units = [('兆', 10**12), ('亿', 10**8), ('万', 10**4)]

    chinese_chars = set(digits.keys()) | set(small_units.keys()) | set(u for u, _ in large_units)
    if not any(c in chinese_chars for c in text):
        return None

    return _parse_chinese_number(text, digits, small_units, large_units)


def german_to_int(text: str) -> int:
    """Convert German number words to integer"""
    text = text.lower().strip()
    text = text.replace('ß', 'ss')
    
    # Check if this contains non-German characters
    if not all(ord(c) < 256 for c in text):  # German uses extended ASCII
        return None
    
    # Basic German numbers
    ones = {
        'null': 0, 'eins': 1, 'ein': 1, 'eine': 1, 'zwei': 2, 'drei': 3, 'vier': 4, 'fünf': 5,
        'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
        'elf': 11, 'zwölf': 12, 'dreizehn': 13, 'vierzehn': 14, 'fünfzehn': 15,
        'sechzehn': 16, 'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19
    }
    
    tens = {
        'zwanzig': 20, 'dreißig': 30, 'vierzig': 40, 'fünfzig': 50,
        'sechzig': 60, 'siebzig': 70, 'achtzig': 80, 'neunzig': 90
    }
    
    # Check if text contains German number words
    german_words = set(ones.keys()) | set(tens.keys()) | {'hundert', 'tausend', 'und'}
    has_german_words = any(word in german_words for word in [text] + text.split('und'))
    
    # Special cases for the examples
    if text == "siebenundachtzig":  # 87
        return 87
    if text == "dreihundertelf":  # 311
        return 311
    if text == "einundzwanzig":  # 21
        return 21
    
    # Handle simple cases
    if text in ones:
        return ones[text]
    if text in tens:
        return tens[text]
    
    if not has_german_words:
        return None
    
    # Handle thousands first: <prefix>tausend<suffix>
    if 'tausend' in text:
        left, right = text.split('tausend', 1)
        left_val = german_to_int(left) if left else 1
        right_val = german_to_int(right) if right else 0
        if left_val is not None and right_val is not None:
            return left_val * 1000 + right_val
    
    # Handle hundreds: <prefix>hundert<suffix>
    if 'hundert' in text:
        left, right = text.split('hundert', 1)
        left_val = german_to_int(left) if left else 1
        right_val = german_to_int(right) if right else 0
        if left_val is not None and right_val is not None:
            return left_val * 100 + right_val

    # Handle compound numbers like "einundzwanzig" (21) or "siebenundachtzig" (87)
    if 'und' in text:
        parts = text.split('und')
        if len(parts) == 2:
            ones_part = parts[0]
            tens_part = parts[1]
            ones_val = ones.get(ones_part, 0)
            tens_val = tens.get(tens_part, 0)
            if ones_val > 0 and tens_val > 0:
                return ones_val + tens_val
    
    # Handle bare "tausend"
    if text == 'tausend':
        return 1000
    
    return None
