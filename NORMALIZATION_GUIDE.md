# Armenian Address Normalization Guide

This document outlines the comprehensive normalization rules used in the Armenian address matching system to standardize addresses from different registries (SPR and Cadastre).

## Overview

The `AddressNormalizer` class applies a series of text transformations to create consistent, comparable address formats. This enables accurate matching between different database representations of the same address.

## Normalization Process Flow

1. **Pre-processing replacements** (before uppercase conversion)
2. **Case normalization** (uppercase conversion)
3. **Pattern-based transformations** (ordinal numbers, suffixes)
4. **Word-level filtering** (unwanted words, one-letter words)
5. **Suffix removal**
6. **Punctuation normalization**
7. **Ending normalization** (Armenian word endings)
8. **Whitespace normalization**

## Detailed Normalization Rules

### 1. Pre-processing Text Replacements

Applied **before** uppercase conversion to handle case-sensitive patterns:

```python
# Armenian conjunctions - standardize to ligature
'ԵՎ' → 'և'
'եվ' → 'և'

# Common abbreviations and variations
'Անտառավան' → 'Անտառ'
'ԱՆՏԱՌԱՎԱՆ' → 'ԱՆՏԱՌ'
'ՀՐ․' → 'ՀՐԱՊԱՐԱԿ'

# Street name variations (handles multiple dash types)
'ՄՈՒՇ[-֊‐‑‒–—―]2' → 'ՄՈՒՇ 2'  # Regex pattern for various dashes
```

**Dash Types Handled:**
- `-` Regular hyphen-minus (U+002D)
- `֊` Armenian hyphen (U+058A)
- `‐` Hyphen (U+2010)
- `‑` Non-breaking hyphen (U+2011)
- `‒` Figure dash (U+2012)
- `–` En dash (U+2013)
- `—` Em dash (U+2014)
- `―` Horizontal bar (U+2015)

### 2. Case Normalization

All text is converted to uppercase for consistent comparison:
```python
text = text.upper()
```

### 3. Ordinal Number Normalization

Armenian ordinal numbers are converted to Arabic numerals:

```python
# Pattern: number + dash + Armenian ordinal suffix
(\d+)[-֊‐‑‒–—―]?(ՐԴ|ՆԴ|ԻՆ|ԱՄ)\.? → \1

# Examples:
'1-ին' → '1'
'2-րդ' → '2' 
'3֊րդ' → '3'
'15-ԱՄ' → '15'
```

**Ordinal Suffixes Handled:**
- `ՐԴ` (rd - for 2nd, 3rd, etc.)
- `ՆԴ` (nd - alternative form)
- `ԻՆ` (st - for 1st)
- `ԱՄ` (th - for numbered items)

### 4. Word Removal Rules

#### Marshal-related Words
```python
# Remove military rank words (case-insensitive)
'ՄԱՐՇԱԼ' → '' (removed)
'ՇԱՐԼ' → '' (removed)  
'ՇԱՌԼ' → '' (removed)
```

#### Single Armenian Letters
```python
# Remove one-letter Armenian words with optional dots
[Ա-Ֆ]\.? → '' (removed)

# Examples:
'Ա.' → ''
'Բ' → ''
'Գ.' → ''
```

### 5. Armenian Suffix Removal

A comprehensive list of Armenian address suffixes are removed:

```python
# Street/location suffixes (case-insensitive, with optional dots)
'ԽՃՂ', 'ԽՃ', 'րդ', 'ին', 'ՆՐԲ', 'նրբ', 'Նրբ'
'ՇԱՐՔ', 'Շարք', 'շարք', 'Անցղ', 'ԱՆՑՂ'
'Փկղ', 'ՓԿՂ', 'ԱՆՑՂ', 'շարք', 'ԲՆ', 'ՃՂ'
'Փ', 'ՊՈՂ', 'Պ', 'ԱՎ', 'ՃԱՄԲ', 'ՏՈՒՆ', 'Տ․'
'Հող', 'ՓԱԿ', 'շարք', 'թղմ', 'ԹՂՄ'
'թաղամասի', 'ԹԵԼԱ', 'ՃԱՆ', 'Շրջ', 'թ/ղ', 'թղմ'
```

### 6. Punctuation Normalization

```python
# Remove all punctuation except forward slashes and hyphens
[^\w\s/\-] → '' (removed)

# Preserved characters: letters, digits, whitespace, /, -
```

### 7. Armenian Word Ending Normalization

Remove common Armenian grammatical endings:

```python
# Remove trailing 'Ի' (genitive case ending)
word + 'Ի' → word

# Remove trailing 'ՅԻ' (possessive ending)  
word + 'ՅԻ' → word

# Remove trailing 'ՈՒ' (another grammatical ending)
word + 'ՈՒ' → word
```

### 8. Final Cleanup

```python
# Normalize multiple whitespaces to single space
\s+ → ' '

# Trim leading and trailing whitespace
text.strip()
```

## Street Name Aliases and Mappings

### Direct Aliases (applied before normalization)
```python
"Խ. ՀԱՅՐԻԿ" → "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ"
"ԽՐԻՄՅԱՆ ՀԱՅՐԻԿ" → "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ"
```

### Historical Street Name Mappings

The system maintains a comprehensive mapping of old Soviet-era street names to modern Armenian names. All keys and values in this mapping are normalized using the `_norm()` function.

**Examples of mappings (after normalization):**
```python
"ՖՐՈՒՆԶԵ" → "Լ ՄԱԴՈՅԱՆ"
"ԼԵՆԻՆԳՐԱԴՅԱՆ" → "Վ ՍԱՐԳՍՅԱՆ"  
"ԿԻՐՈՎԱԿԱՆՅԱՆ" → "ՎԱՆԱՁՈՐ"
"ԿԱԼԻՆԻՆ" → "Գ ՆԺԴԵՀ"
"ԼԵՆԻՆ" → "ՏԻԳՐԱՆ ԶԱ"
"ՄԱՐՔՍ" → "Պ ՋԱՓԱՐԻՁԵ"
# ... and 100+ more mappings
```

## Number Part Normalization

For house numbers and building numbers, a simplified normalization is applied:

```python
def normalize_number_part(text):
    # Convert to uppercase
    text = text.upper()
    
    # Apply ordinal number normalization
    (\d+)[-֊‐‑‒–—―]?(ՐԴ|ՆԴ|ԻՆ|ԱՄ)\.? → \1
    
    # Remove all non-word characters except / and -
    [^\w/\-] → ''
    
    return text
```

## Full Address Construction

Addresses are constructed by concatenating normalized components:

```python
FULL_ADDRESS = STREET_NORM + " " + SUB_STREET_NORM + " " + HOUSE_NORM + " " + BUILDING_NORM
```

- Multiple spaces are collapsed to single spaces
- Leading and trailing whitespace is removed

## Data Quality Metrics

The system calculates completeness scores:

**For SPR Registry:**
```python
COMPLETENESS_SCORE = (has_street + has_house + has_building) / 3
```

**For Cadastre Registry:**
```python  
COMPLETENESS_SCORE = (has_street + has_sub_street + has_house + has_building) / 4
```

## Matching Strategy

1. **Exact Full Address Matching**: Direct comparison of normalized full addresses
2. **Component-based Matching**: Street + house number combinations
3. **Fuzzy Matching**: For near-matches with acceptable similarity scores

## Unicode Considerations

The system properly handles:
- Armenian Unicode characters (U+0530 to U+058F)
- Various dash/hyphen Unicode variants
- Armenian ligatures (և)
- Case-sensitive vs case-insensitive operations

## Performance Optimizations

- **Indexing**: Street-based and house-based indices for fast lookups
- **Chunked Processing**: Large datasets processed in chunks of 500 records
- **Caching**: Normalized mappings cached for repeated use

## Usage Notes

1. All normalization is designed to be **idempotent** - applying it multiple times produces the same result
2. The order of operations matters - pre-processing must occur before uppercase conversion
3. The system handles missing/null values by converting them to empty strings
4. Progress tracking and user interface updates are integrated for long-running operations

## Common Pitfalls

1. **Dash Character Issues**: Always use regex patterns for dash handling rather than single character replacements
2. **Case Sensitivity**: Apply case-sensitive replacements before uppercase conversion
3. **Unicode Normalization**: Ensure proper handling of Armenian Unicode characters
4. **Whitespace**: Always normalize whitespace as the final step

This normalization system enables robust matching between address databases with different formatting conventions, historical naming variations, and encoding inconsistencies.