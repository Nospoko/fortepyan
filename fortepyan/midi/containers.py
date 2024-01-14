import re


class Instrument(object):
    """Object to hold event information for a single instrument.

    Parameters:
        program (int): MIDI program number (instrument index), in ``[0, 127]``.
        is_drum (bool, optinal): Is the instrument a drum instrument (channel 9)?
        name (str, optional): Name of the instrument.

    Notes:
        It's a container class used to store notes, and control changes. Adapted from [pretty_midi](https://github.com/craffel/pretty-midi).

    """

    def __init__(self, program, is_drum=False, name=""):
        self.program = program
        self.is_drum = is_drum
        self.name = name
        self.pitch_bends = []
        self.notes = []
        self.control_changes = []

    def get_end_time(self):
        """Returns the time of the end of the events in this instrument.

        Returns
        -------
        end_time : float
            Time, in seconds, of the last event.

        """
        # Cycle through all note ends and all pitch bends and find the largest
        events = [n.end for n in self.notes] + [c.time for c in self.control_changes]
        # If there are no events, just return 0
        if len(events) == 0:
            return 0.0
        else:
            return max(events)


class Note(object):
    """A note event.

    Parameters:
        velocity (int): Note velocity.
        pitch (int): Note pitch, as a MIDI note number.
        start (float): Note on time, absolute, in seconds.
        end (float): Note off time, absolute, in seconds.

    Notes:
        It's a container class used to store a note. Adapted from [pretty_midi](https://github.com/craffel/pretty-midi).

    """

    def __init__(self, velocity, pitch, start, end):
        if end < start:
            raise ValueError("Note end time must be greater than start time")

        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end

    def get_duration(self):
        """
        Get the duration of the note in seconds.
        """
        return self.end - self.start

    @property
    def duration(self):
        return self.get_duration()

    def __repr__(self):
        return "Note(start={:f}, end={:f}, pitch={}, velocity={})".format(self.start, self.end, self.pitch, self.velocity)


class ControlChange(object):
    """
    A control change event.

    Parameters:
        number (int): The control change number, in ``[0, 127]``.
        value (int): The value of the control change, in ``[0, 127]``.
        time (float): Time where the control change occurs.

    Notes:
        It's a container class used to store a control change. Adapted from [pretty_midi](https://github.com/craffel/pretty-midi).
    """

    def __init__(self, number, value, time):
        self.number = number
        self.value = value
        self.time = time

    def __repr__(self):
        return "ControlChange(number={:d}, value={:d}, " "time={:f})".format(self.number, self.value, self.time)


class KeySignature(object):
    """Contains the key signature and the event time in seconds.
    Only supports major and minor keys.

    Attributes:
        key_number (int): Key number according to ``[0, 11]`` Major, ``[12, 23]`` minor.
        For example, 0 is C Major, 12 is C minor.
        time (float): Time of event in seconds.

    Example:
    Instantiate a C# minor KeySignature object at 3.14 seconds:

    >>> ks = KeySignature(13, 3.14)
    >>> print(ks)
    C# minor at 3.14 seconds
    """

    def __init__(self, key_number, time):
        if not all((isinstance(key_number, int), key_number >= 0, key_number < 24)):
            raise ValueError("{} is not a valid `key_number` type or value".format(key_number))
        if not (isinstance(time, (int, float)) and time >= 0):
            raise ValueError("{} is not a valid `time` type or value".format(time))

        self.key_number = key_number
        self.time = time

    def __repr__(self):
        return "KeySignature(key_number={}, time={})".format(self.key_number, self.time)

    def __str__(self):
        return "{} at {:.2f} seconds".format(key_number_to_key_name(self.key_number), self.time)


class Lyric(object):
    """
    Timestamped lyric text.

    """

    def __init__(self, text, time):
        self.text = text
        self.time = time

    def __repr__(self):
        return 'Lyric(text="{}", time={})'.format(self.text.replace('"', r"\""), self.time)

    def __str__(self):
        return '"{}" at {:.2f} seconds'.format(self.text, self.time)


class Text(object):
    """
    Timestamped text event.
    """

    def __init__(self, text, time):
        self.text = text
        self.time = time

    def __repr__(self):
        return 'Text(text="{}", time={})'.format(self.text.replace('"', r"\""), self.time)

    def __str__(self):
        return '"{}" at {:.2f} seconds'.format(self.text, self.time)


def key_number_to_key_name(key_number):
    """
    Convert a key number to a key string.

    Parameters:
        key_number (int): Uses pitch classes to represent major and minor keys. For minor keys, adds a 12 offset. For example, C major is 0 and C minor is 12.

    Returns:
    key_name (str): Key name in the format ``'(root) (mode)'``, e.g. ``'Gb minor'``. Gives preference for keys with flats, with the exception of F#, G# and C# minor.
    """

    if not isinstance(key_number, int):
        raise ValueError("`key_number` is not int!")
    if not ((key_number >= 0) and (key_number < 24)):
        raise ValueError("`key_number` is larger than 24")

    # preference to keys with flats
    keys = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    # circle around 12 pitch classes
    key_idx = key_number % 12
    mode = key_number // 12

    # check if mode is major or minor
    if mode == 0:
        return keys[key_idx] + " Major"
    elif mode == 1:
        # preference to C#, F# and G# minor
        if key_idx in [1, 6, 8]:
            return keys[key_idx - 1] + "# minor"
        else:
            return keys[key_idx] + " minor"


def key_name_to_key_number(key_string):
    """
    Convert a key name string to key number.

    Parameters:
        key_string (str): Format is ``'(root) (mode)'``, where:
          * ``(root)`` is one of ABCDEFG or abcdefg.  A lowercase root
            indicates a minor key when no mode string is specified.  Optionally
            a # for sharp or b for flat can be specified.

          * ``(mode)`` is optionally specified either as one of 'M', 'Maj',
            'Major', 'maj', or 'major' for major or 'm', 'Min', 'Minor', 'min',
            'minor' for minor.  If no mode is specified and the root is
            uppercase, the mode is assumed to be major; if the root is
            lowercase, the mode is assumed to be minor.

    Returns:
        key_number (int):
            Integer representing the key and its mode.  Integers from 0 to 11
            represent major keys from C to B; 12 to 23 represent minor keys from C
            to B.
    """
    # Create lists of possible mode names (major or minor)
    major_strs = ["M", "Maj", "Major", "maj", "major"]
    minor_strs = ["m", "Min", "Minor", "min", "minor"]
    # Construct regular expression for matching key
    pattern = re.compile(
        # Start with any of A-G, a-g
        "^(?P<key>[ABCDEFGabcdefg])"
        # Next, look for #, b, or nothing
        "(?P<flatsharp>[#b]?)"
        # Allow for a space between key and mode
        " ?"
        # Next, look for any of the mode strings
        "(?P<mode>(?:(?:"
        +
        # Next, look for any of the major or minor mode strings
        ")|(?:".join(major_strs + minor_strs)
        + "))?)$"
    )
    # Match provided key string
    result = re.match(pattern, key_string)
    if result is None:
        raise ValueError("Supplied key {} is not valid.".format(key_string))
    # Convert result to dictionary
    result = result.groupdict()

    # Map from key string to pitch class number
    key_number = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[result["key"].lower()]
    # Increment or decrement pitch class if a flat or sharp was specified
    if result["flatsharp"]:
        if result["flatsharp"] == "#":
            key_number += 1
        elif result["flatsharp"] == "b":
            key_number -= 1
    # Circle around 12 pitch classes
    key_number = key_number % 12
    # Offset if mode is minor, or the key name is lowercase
    if result["mode"] in minor_strs or (result["key"].islower() and result["mode"] not in major_strs):
        key_number += 12

    return key_number


class TimeSignature(object):
    """
    Container for a Time Signature event, which contains the time signature
    numerator, denominator and the event time in seconds.

    Attributes:
        numerator (int):
            Numerator of time signature.
        denominator (int):
            Denominator of time signature.
        time (float):
            Time of event in seconds.

    Example:
        Instantiate a TimeSignature object with 6/8 time signature at 3.14 seconds:

        >>> ts = TimeSignature(6, 8, 3.14)
        >>> print(ts)
        6/8 at 3.14 seconds
    """

    def __init__(self, numerator, denominator, time):
        if not (isinstance(numerator, int) and numerator > 0):
            raise ValueError("{} is not a valid `numerator` type or value".format(numerator))
        if not (isinstance(denominator, int) and denominator > 0):
            raise ValueError("{} is not a valid `denominator` type or value".format(denominator))
        if not (isinstance(time, (int, float)) and time >= 0):
            raise ValueError("{} is not a valid `time` type or value".format(time))

        self.numerator = numerator
        self.denominator = denominator
        self.time = time

    def __repr__(self):
        return "TimeSignature(numerator={}, denominator={}, time={})".format(self.numerator, self.denominator, self.time)

    def __str__(self):
        return "{}/{} at {:.2f} seconds".format(self.numerator, self.denominator, self.time)


class PitchBend(object):
    """
    A pitch bend event.

    Parameters:
        pitch (int)
            MIDI pitch bend amount, in the range ``[-8192, 8191]``.
        time (float)
            Time where the pitch bend occurs.

    """

    def __init__(self, pitch, time):
        self.pitch = pitch
        self.time = time

    def __repr__(self):
        return "PitchBend(pitch={:d}, time={:f})".format(self.pitch, self.time)
