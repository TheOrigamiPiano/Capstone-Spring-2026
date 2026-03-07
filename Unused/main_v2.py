from collections import Counter
from fractions import Fraction
from typing import Dict

import music21
from music21 import converter, note, midi, duration, meter, interval, pitch
from music21.interval import GenericInterval
from music21.meter import TimeSignature
from music21.midi import MidiFile
from music21.note import GeneralNote
from music21.stream import Measure
from music21.stream.makeNotation import consolidateCompletedTuplets

import numpy as np

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


