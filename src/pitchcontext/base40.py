"""Provides a mapping from base40 pitch encoding to textual pitch representation. For one octave only."""

base40 = {
    0:"C--",
    1:"C-",
    2:"C",
    3:"C#",
    4:"C##",
    5:"D---",
    6:"D--",
    7:"D-",
    8:"D",
    9:"D#",
    10:"D##",
    11:"E---",
    12:"E--",
    13:"E-",
    14:"E",
    15:"E#",
    16:"E##",
    17:"F--",
    18:"F-",
    19:"F",
    20:"F#",
    21:"F##",
    22:"unused",
    23:"G--",
    24:"G-",
    25:"G",
    26:"G#",
    27:"G##",
    28:"A---",
    29:"A--",
    30:"A-",
    31:"A",
    32:"A#",
    33:"A##",
    34:"B---",
    35:"B--",
    36:"B-",
    37:"B",
    38:"B#",
    39:"B##",
}

#As list
base40list = [base40[key] for key in range(0,40)]

#Only natural tones
sel=['A','B','C','D','E','F','G']
base40naturalslist = [base40[key] if base40[key] in sel else '' for key in range(0,40)]
