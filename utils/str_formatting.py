from math import floor


def short_number(n: int) -> str:
    """
    Convert an integer into a short string notation using 'k' for 1 000 and 'M' for 1 000 000.
    Args:
        n: The integer to format.
    Returns:
        The formatted string.
    """
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f'{n / 1_000:.1f}'.rstrip('0').rstrip('.') + 'k'  # Remove unnecessary decimal 0
    # >= 1_000_000
    return f'{n / 1_000_000:.1f}'.rstrip('0').rstrip('.') + 'M'  # Remove unnecessary decimal 0


def duration_to_str(sec: float, nb_units_display: int = 2, precision: str = 'ms'):
    """
    Transform a duration (in sec) into a human readable string.
    d: day, h: hour, m: minute, s: second, ms: millisecond

    :param sec: The number of second of the duration. Decimals are milliseconds.
    :param nb_units_display: The maximum number of unit we want to show. If 0 print all units.
    :param precision: The smallest unit we want to show.
    :return: A human-readable representation of the duration.
    """

    assert sec >= 0, f'Negative duration not supported ({sec})'
    assert nb_units_display > 0, 'At least one unit should be displayed'
    precision = precision.strip().lower()
    assert precision in ['d', 'h', 'm', 's', 'ms'], 'Precision should be a valid unit: d, h, m, s, ms'

    # Null duration
    if sec == 0:
        return '0' + precision

    # Infinite duration
    if sec == float('inf'):
        return 'infinity'

    # Convert to ms
    mills = floor(sec * 1_000)

    periods = [
        ('d', 1_000 * 60 * 60 * 24),
        ('h', 1_000 * 60 * 60),
        ('m', 1_000 * 60),
        ('s', 1_000),
        ('ms', 1)
    ]

    strings = []
    for period_name, period_mills in periods:
        if mills >= period_mills:
            period_value, mills = divmod(mills, period_mills)
            strings.append(f"{period_value}{period_name}")
        # Stop if we reach the minimal precision unit
        if period_name == precision:
            if len(strings) == 0:
                return '<1' + period_name
            break

    if nb_units_display > 0:
        strings = strings[:nb_units_display]

    return ' '.join(strings)
