import os
from collections import Counter
from fractions import Fraction
from typing import Dict

import matplotlib
import music21
from matplotlib import pyplot as plt
from music21 import converter, note, midi, duration, meter, interval, pitch
from music21.interval import GenericInterval
from music21.meter import TimeSignature
from music21.midi import MidiFile
from music21.note import GeneralNote
from music21.stream import Measure
from music21.stream.makeNotation import consolidateCompletedTuplets

import numpy as np

from src.motif_finder.main import create_song_object


# New Process:
# 1) Use the string-join approach to get an RP-tree of the piece (uses SimpleNote objects)
#   a) Along the way, save every section of song that does not repeat.
# 2) Traverse the RP-tree to create a list in order by position
#   a) Add back in the sections that don't repeat, and format them in the same way as MusicString objects
#   b) Add back in all rests? (to-fix: Figure this out)
# 3) Convert SimpleNote to SimpleNotePrime to create a MusicStringPrime list
#   a) In each MusicStringPrime object, make sure to correctly add the ending prime object, which leads out of the
#   repeated phrase. (the last phrase will not have one)
# 4) Create a new map object, PhraseMap. The key is a new object, PhraseKey, which stores the first appearance as a
# PhrasePosition object, and an int for the number of songs this phrase appears in.
# The value will be a list of MusicString objects, which make up all similar phrases within that group.
# 5 Compare and merge phrases within the MusicStringPrime list
#   a) Compare phrases of same length to see if they are the same, because the new phrase might be the original but
#   transposed. If so, combine the objects into the first by updating the position list.
#   b) Take each repeated phrase, and try to add it with the phrase after it. Compare these new phrases using a custom
#   algorithm.
#     i) If the custom algorithm returns that the two are similar, then
#

# Reduces a note to its pitch and duration (in quarter notes)
class SimpleNote(object):
    pitch: int = 0
    note_duration: float = 0.0
    tie: str | None = None
    
    # Constructor
    def __init__(self, pitch, note_duration, tie):
        self.pitch = pitch
        self.note_duration = note_duration
        self.tie = tie
    
    def __str__(self):
        return "pitch: {0}, note_duration: {1}".format(
            self.pitch, self.note_duration
        )


class SimpleNotePrime(object):
    generic_interval: int = 0
    duration_ratio: float = 0.0
    
    # Constructor
    def __init__(self, generic_interval, duration_ratio):
        self.generic_interval = generic_interval
        self.duration_ratio = duration_ratio
    
    def __str__(self):
        return "generic_interval: {0}, duration_ratio: {1}".format(
            self.generic_interval, self.duration_ratio
        )
    
    def __hash__(self):
        return hash((self.generic_interval, self.duration_ratio))
    
    def __eq__(self, other):
        return self.generic_interval == other.generic_interval and self.duration_ratio == other.onset_ratio
    

# Marks a phrase's start position in terms of song number, track, measure, and offset of measure
class PhrasePosition(object):
    song_index: int
    track: str
    measure_number: int
    offset: float
    
    def __init__(self, song_index, track, measure_number, offset):
        self.song_index = song_index
        self.track = track
        self.measure_number = measure_number
        self.offset = offset
        
# The key that will mark a certain phrase group
class PhraseKey(object):
    first_position: PhrasePosition
    number_of_songs: int  # Number of songs this phrase group appears in


# Represents a nontrivial music repetition of len(note_list)
class MusicString(object):
    note_list: list[SimpleNote]
    frequency: int  # Number of exact repetitions of this phrase
    positions: list[PhrasePosition]  # Position of exact repetitions
    
    def __init__(self, note_list, frequency, positions):
        self.note_list = note_list
        self.frequency = frequency
        self.positions = positions



# Flattens a list of lists into a singular list
def flatten_list(measure_list):
    flat_list = []
    for measure in measure_list:
        flat_list.extend(measure)
    return flat_list

def calculate_dice_coefficient(measure_prime_1: list[SimpleNotePrime], measure_prime_2: list[SimpleNotePrime]):
    intersection_set = list((Counter(measure_prime_1) & Counter(measure_prime_2)).elements())
    
    if len(measure_prime_1) == 0 or len(measure_prime_2) == 0:
        return 0.0
    else:
        return 2 * len(intersection_set) / (len(measure_prime_1) + len(measure_prime_2))


# Takes two measure_prime objects and returns a similarity value between 0 and 1.
# (Can switch between different available algorithms)
def measure_of_similarity(measure_prime_1: list[SimpleNotePrime], measure_prime_2: list[SimpleNotePrime]):
    value = calculate_dice_coefficient(measure_prime_1, measure_prime_2)
    return value


def create_self_similarity_matrix(measures_prime: list[list[SimpleNotePrime]]) -> list[list[float]]:
    length = len(measures_prime)
    self_similarity_matrix = [[0.0 for i in range(length)] for j in range(length)]
    for index1, measure1 in enumerate(measures_prime):
        for index2, measure2 in enumerate(measures_prime):
            self_similarity_matrix[index1][index2] = calculate_dice_coefficient(measure1, measure2)
    return self_similarity_matrix


def create_boolean_ssm(self_similarity_matrix: list[list[float]], threshold: float) -> list[list[bool]]:
    boolean_ssm: list[list[bool]] = []
    for row in self_similarity_matrix:
        boolean_ssm.append([bool(x > threshold) for x in row])
    return boolean_ssm


# Create a lag matrix, which is a representation of a ssm where the diagonals are turned into rows
# Only uses the bottom-left half of the ssm, since the other half is repeat information
# TO-DO: Delete later
def create_lag_matrix(boolean_ssm: list[list[bool]]):
    size = len(boolean_ssm)
    lag_matrix = [[False for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            if i + j < size:
                lag_matrix[i][j] = boolean_ssm[i + j][j]
            else:
                break
    return lag_matrix

# def traverse_boolean_ssm(boolean_ssm: list[list[bool]]):


def process_midi_file(midi_filepath: str, song_name: str, song_index: int):
    # Create song object
    song = create_song_object(midi_filepath, song_name, song_index)
    
    # Print self-similarity matrix for each part
    track_ssm_list: list[list[list[float]]] = []
    for part in list(song.sky_prime_notes_data.keys()):
        self_similarity_matrix = create_self_similarity_matrix(song.sky_prime_notes_data[part])
        track_ssm_list.append(self_similarity_matrix)
    
    threshold: float = 0.7
    boolean_ssm_list: list[list[list[bool]]] = []
    for ssm in track_ssm_list:
        boolean_ssm = create_boolean_ssm(ssm, threshold)
        boolean_ssm_list.append(boolean_ssm)
    
    lag_matrix_list: list[list[list[bool]]] = []
    for boolean_ssm in boolean_ssm_list:
        lag_matrix = create_lag_matrix(boolean_ssm)
        lag_matrix_list.append(lag_matrix)
    
    song_name = os.path.basename(midi_filepath)
# - Uncomment to see SSM and Lag Matrix graphs
# song_name = "Space Junk Road"
# Draw First Graphs
# first_track = list(simple_midi_prime_data.keys())[0]
# plot_colored_grid(boolean_ssm_list[0], song_name, first_track)
# plot_colored_grid(lag_matrix_list[0], song_name, first_track)

# Draw All Graphs
# for index, part in enumerate(list(simple_midi_prime_data.keys())):
#     plot_colored_grid(boolean_ssm_list[index], song_name, part)
#     plot_colored_grid(lag_matrix_list[index], song_name, part)

# return simple_midi_data, lag_matrix_list

def test_single_file():
    midi_file = "../../MidiFiles/superMarioGalaxy/Super Mario Galaxy - Rosalinas Comet Observatory 1 2  3.mid"
    process_midi_file(midi_file)


def test_multiple_files():
    folder_name = "MidiFiles/superMarioGalaxy"
    midi_files = []
    for (dirpath, dirnames, filenames) in os.walk(folder_name):
        midi_files.extend(filenames)
    
    for midi_file in midi_files:
        process_midi_file(folder_name + "/" + midi_file)


def plot_colored_grid(data, song_name, part_name):
    # Color for False and True
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])
    
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 200
    plt.imshow(data, cmap=cmap)
    plt.title("{0}: {1}".format(song_name, part_name))
    plt.show()