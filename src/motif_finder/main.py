import dataclasses
from dataclasses import dataclass
import json
import os
from collections import Counter
from fractions import Fraction
from typing import Dict
from xml.sax import default_parser_list

import music21
from music21 import converter, note, midi, duration, meter, interval, pitch, chord
from music21.interval import GenericInterval
from music21.meter import TimeSignature
from music21.midi import MidiFile
from music21.note import GeneralNote
from music21.stream import Measure, Score
from music21.stream.iterator import StreamIterator
from music21.stream.makeNotation import consolidateCompletedTuplets

from librosa import segment

import matplotlib.pyplot as plt
import matplotlib.colors

import numpy as np
from music21.tree.toStream import chordified
from numba.typed.dictobject import new_dict
from scipy.signal import freqs

# Troubleshooting notes for correctly reading MIDI files
# - In some songs, notes are present but not properly stored in Measure objects? Either way, score.recurse().notes will
#   work instead.
#
# To-Fix
# - Most songs need remade measures, but this breaks for Rosalina's observatory, because (for some reason) incorrect
#   midi formatting creates extra measures. For this song, the applicable test results used remake_measures = False.
#
# To-Do
# - Add more midi files and tests (both self-compare and cross-compare)
# - Fix skyline algorithm to properly differentiate single parts with multiple voices
#   - Use Dark Sanctuary as a guide, since I think it pushes every rule for this function
# - Fix query_similar_skyline_leitmotif() function to not identify only part of a motif and then skip over it
#   - Alternatively, modify the Smith-Waterman algorithm to take the entire song as one input and then find all motifs
#     concurrently.
#
# Future Goals
# - Develop process for automatically identifying possible motifs, which will also build off the Smith-Waterman algorithm
# - If the entire song repeats, cut it in half
# - Create a script for automatically loading midi files into Musescore and re-downloading it (to fix formatting errors)
#
# To-Review
# - Review what objects are actually needed for the song class
# - OriginalNotes could  be consolidated as SimpleNotes (or vice versa, depending on how I want to code chords)


# Notes
# - .exec method for running string as code

@dataclass
class OriginalNote(object):
    pitches: list[int]
    duration: float = 0.0
    measure_number: int = 0
    total_onset: float = 0.0
    local_onset: float = 0.0
    tie: str | None = None
    
    def short_repr(self):
        return "({0}, {1})".format(self.pitches, self.duration)

@dataclass
class SimpleNote(object):
    pitch: int = 0
    duration: float = 0.0
    onset: float = 0.0
    interonset_interval: float | None = None  #Onset difference between this note and the next
    measure_number: int = 0
    note_measure_index: int = 0  #Index of the note within a measure (similar to offset)
    tie: str | None = None
    
    def short_repr(self):
        return "({0}, {1})".format(self.pitch, self.duration)
    
@dataclass
class SimpleNotePrime(object):
    generic_interval: int = 0
    onset_ratio: float = 0.0
    measure_number: int = 0
    note_measure_index: int = 0  #Note: Measure number and note index are in relation to the "first" note of each pair

    def compare_note(self, other):
        return self.generic_interval == other.generic_interval and self.onset_ratio == other.onset_ratio


@dataclass
class Song(object):
    song_index: int
    song_name: str
    original_notes_data: dict[str, list[list[OriginalNote]]]  #Full, original sequence grouped by measure
    simple_notes_data: dict[str, list[list[SimpleNote]]]  #All notes without rests, grouped by chord (not measure)
    prime_notes_data: dict[str, list[list[list[SimpleNotePrime]]]]  #All prime note combinations, grouped by chord
    sky_simple_notes_data: dict[str, list[SimpleNote]]  #Skyline notes without rests
    sky_prime_notes_data: dict[str, list[SimpleNotePrime]]  #Skyline prime notes without rests
    
    def get_parts_list(self):
        return list(self.original_notes_data.keys())
    
    def print_simple_notes_data(self):
        for part_name, simple_notes_list in self.simple_notes_data.items():
            print(part_name)
            print_simple_notes(simple_notes_list)
            
    def print_prime_notes_data(self):
        for part_name, prime_notes_list in self.prime_notes_data.items():
            print(part_name)
            print_prime_notes(prime_notes_list)
            
    def print_sky_simple_notes_data(self):
        for part_name, sky_notes in self.sky_simple_notes_data.items():
            print(part_name)
            print_sky_notes(sky_notes)
            
    def print_sky_prime_notes_data(self):
        for part_name, sky_notes in self.sky_prime_notes_data.items():
            print(part_name)
            print_sky_notes(sky_notes)
            
    # Incomplete: Test out new print function to easily organize and compact simple note information
    # Used for troubleshooting any midi formatting issues
    # def nice_print_simple_notes_data(self):
    #     for part_name, simple_notes_list in self.simple_notes_data.items():
    
    


# Marks a phrase's start position in terms of song name, part name, measure, and note index of measure
@dataclass
class PhrasePosition(object):
    song_name: str
    part_name: str
    measure_number: int
    note_measure_index: int


# Represents a nontrivial music repetition of len(note_prime_list)
@dataclass
class MusicPhrase(object):
    prime_notes: list[SimpleNotePrime]
    frequency: int  #Number of exact repetitions of this phrase
    positions: list[PhrasePosition]  #Position of exact repetitions
    
    def update(self, position: PhrasePosition):
        self.frequency += 1
        self.positions.append(position)
        
    def get_first_position(self):
        return self.positions[0]
        
    
# A container for the music_string_list
@dataclass
class PhraseGroup(object):
    music_phrase_list: list[MusicPhrase]
    
    # Internal variables (don't need to be added with object instantiation)
    song_names: list[str]  #List of songs this phrase group appears in
    
    def __init__(self, music_string_list: list[MusicPhrase]):
        self.music_phrase_list = music_string_list
        self.song_names = []
    
    # Always use this method to add a phrase to a phrase_group object
    def add(self, music_string: MusicPhrase):
        self.music_phrase_list.append(music_string)
        
        if music_string.get_first_position().song_name not in self.song_names:
            song_name = music_string.get_first_position().song_name
            self.song_names.append(song_name)
    
    # Original phrase is defined as the first time the phrase appears
    def get_original_phrase(self):
        return self.music_phrase_list[0]
    
    
# Global Variables
song_dict: dict[str, Song] = {}
phrase_group_list: list[PhraseGroup] = []
total_song_length: float = 0

# Modifiable Parameters
min_motif_length: int = 5

def midi_to_notes_by_measure(midi_path, remake_measures=False):
    """
    Extracts notes and rests from a MIDI file organized by track (Part) and by measure

    :param str midi_path: path to MIDI file
    :param bool remake_measures: specify if this function should remake measures according to the main time signature
    (in case time signature information is consistent throughout the piece but is incorrectly updated across the MIDI
    parts). Defaults to false.
    :return: a dictionary where each key is the track name and the value is a list of Measure objects
    """
    
    # Parse MIDI file
    score = converter.parse(midi_path)  # converter.parse(midi_path,  quarterLengthDivisors=(2, 3, 4, 8))
    
    # Prepare dictionary to store information
    midi_data = {}
    
    # Time Signature is initialized by first instrument
    time_signature: TimeSignature | None = None
    
    for i, part in enumerate(score.parts):
        # Get track name from midi. If it doesn't have one, assign it based on the midi instrument
        part_name: str
        if part.partName and not part.partName.__contains__("Track"):
            part_name = part.partName
        else:
            part_name = part[music21.instrument.Instrument][0].__str__()
        
        if i == 0:
            time_signature = part[music21.meter.TimeSignature][0]
        else:
            part['Measure'][0].timeSignature = time_signature
        
        # If applicable, remake measures to account for new time signature
        if remake_measures:
            part = part.stripTies()
            part = part.makeMeasures()
            part = part.makeTies()
            
        # Get measure information
        measure_offset_list = list(part.measureOffsetMap().keys())
        
        # Check for repetitive parts with same part_name, and rename them (for example, piano sometimes has two parts)
        same_track_number = 1
        while part_name in midi_data:
            same_track_number += 1
            part_name = part_name + " " + same_track_number.__str__()
        
        # Recurse notes and further organize notes by measure
        organized_notes: list[list[OriginalNote]] = []
        notes_iterator: StreamIterator[GeneralNote] = part.recurse().notesAndRests
        for note in notes_iterator:
            if not note.isRest:
                pitches = [p.midi for p in note.pitches]
                pitches.sort()
                    
                original_note = OriginalNote(pitches=pitches,
                                             duration=note.quarterLength,
                                             measure_number=note.measureNumber,
                                             local_onset=note.offset,
                                             total_onset=note.offset + measure_offset_list[note.measureNumber - 1])
                
                while len(organized_notes) < note.measureNumber:
                    organized_notes.append([])
                organized_notes[note.measureNumber - 1].append(original_note)
        
        # for measure in organized_notes:
        #     combine_into_chords(measure)
        
        midi_data[part_name] = organized_notes
    
    return midi_data

def combine_into_chords(measure):
    """
    Music21 will divide notes with the same offset into different chords if they have different durations or velocities.
    This function will combine notes into chords regardless of velocity. It will also combine notes into chords
    regardless of duration, reducing the longer note into the shorter one, therefore combining all distinct voices into
    one voice. This simplifies the musical content, but it shouldn't cause any issues. All of these transformations
    occur in place within the measure object.
    
    :param list[OriginalNote] measure: the measure to sift through and combine the applicable notes into chords
    """
    
    index = 0
    for current_note in measure:
        for check_index in range(index):
            prev_note = measure[check_index]
            if current_note.total_onset == prev_note.total_onset:
                # Figure out which of the two is the shorter or longer note
                if current_note.duration < prev_note.duration:
                    shorter_note = current_note
                    
                    longer_note = prev_note
                    index_to_remove = check_index
                else:
                    shorter_note = prev_note
                    
                    longer_note = current_note
                    index_to_remove = index
                
                # Replace and remove respective notes
                new_pitches = shorter_note.pitches + longer_note.pitches
                new_pitches.sort()
                shorter_note.pitches = new_pitches
                measure.pop(index_to_remove)
                
                # Decrease index to account for removed note
                index -= 1
                break
                
        # Increment index
        index += 1


def notes_by_measure_to_simple_notes(notes_by_measure):
    """
    Inputs measures and outputs both simple_notes and sky_simple_notes. For both outputs, all rests are removed.
    Measure number and offset are preserved by individual SimpleNote objects.

    :param list[list[OriginalNote]] notes_by_measure: the original list of measures
    :return: a tuple containing the simple_notes followed by the sky_simple_notes
    """
    
    simple_notes: list[list[SimpleNote]] = []
    sky_simple_notes: list[SimpleNote] = []
    
    for measure in notes_by_measure:
        # Within a measure, identity the melody by the highest note currently being played (skyline algorithm)
        # Keep offset of skyline notes to avoid overlap
        local_sky_offset = 0
        note_measure_index = 0
        for original_note in measure:
            # To-do: Identify if general note is a tied note or not
            
            # Temp: Ties seem to be broken in some midi files, so I'll try skipping all tie continuations.
            # if tie_type.__eq__('Stop'):
            #     continue
            
            # Make a SimpleNote object for each pitch in a chord. The sky note will always be the last element of the
            # chord, since the pitches were previously sorted in ascending order
            simple_note_chord: list[SimpleNote] = []
            for pitch in original_note.pitches:
                simple_note = SimpleNote(pitch=pitch,
                                         duration=original_note.duration,
                                         onset=original_note.total_onset,
                                         measure_number=original_note.measure_number,
                                         note_measure_index=note_measure_index)
                simple_note_chord.append(simple_note)
            
            sky_simple_note = simple_note_chord[len(simple_note_chord) - 1]
            
            # Append sky_simple_note if there is no overlap
            if original_note.local_onset >= local_sky_offset:
                sky_simple_notes.append(sky_simple_note)
                
                # Adjust offset and then add duration
                local_sky_offset = original_note.local_onset + original_note.duration
            
            # Append simple_note_chord
            simple_notes.append(simple_note_chord)
            
            # Increment note_measure_index
            note_measure_index += 1
    
    # Reiterate through lists to define ioi
    for index in range(len(sky_simple_notes) - 1):
        sky_simple_notes[index].interonset_interval = sky_simple_notes[index + 1].onset - sky_simple_notes[index].onset
    
    for chord_index in range(len(simple_notes) - 1):
        for note in simple_notes[chord_index]:
            check = note.duration + note.onset
            temp_index = chord_index + 1
            
            while temp_index < len(simple_notes) and check > simple_notes[temp_index][0].onset:
                temp_index += 1
            
            # Don't assign ioi if there is no next note in the sequence
            if temp_index < len(simple_notes):
                note.interonset_interval = simple_notes[temp_index][0].onset - note.onset
    
    return simple_notes, sky_simple_notes


def simples_sky_notes_to_prime_sky_notes(simple_sky_notes):
    """
    Transforms a list of sky notes into a list of prime sky notes, returning it as a new object.
    
    :param list[SimpleNote] simple_sky_notes: the list of simple sky notes
    :return: a list of prime sky notes
    """
    prime_sky_notes: list[SimpleNotePrime] = []
    
    for index in range(len(simple_sky_notes) - 1):
        current_note = simple_sky_notes[index]
        next_note = simple_sky_notes[index + 1]
        prime_sky_notes.append(find_prime(current_note, next_note))
        
    return prime_sky_notes
    
def simple_notes_to_prime_notes(simple_notes):
    """
    Transforms a collection of note objects into a collection of prime note objects.
    
    :param list[list[SimpleNote]] simple_notes: all simple notes within the collection
    :return: a collection of prime note objects. Order of lists from innermost to outermost goes: prime combinations for
    one note to all notes within the next chord -> prime combinations for entire chord to the entire next chord -> all
    sequential chord information for the entire song
    """
    # Each note in simple_notes is made into a list of all its possible prime connections
    prime_notes: list[list[list[SimpleNotePrime]]] = []
    
    for index in range(len(simple_notes) - 1):
        # Note: chords contain at least one note
        current_chord = simple_notes[index]
        next_chord = simple_notes[index + 1]
        
        chord_connections: list[list[SimpleNotePrime]] = []
        for current_note in current_chord:
            note_connections: list[SimpleNotePrime] = []
            for next_note in next_chord:
                note_connections.append(find_prime(current_note, next_note))
            chord_connections.append(note_connections)
            
        prime_notes.append(chord_connections)
        
    return prime_notes
    
    
def find_prime(current_note, next_note):
    """
    Calculate the pitch interval and IOI ratio between two sequential notes, and then return the respective
    SimpleNotePrime object.
    
    :param SimpleNote current_note: The current note
    :param SimpleNote next_note: The next note
    :return: The SimpleNotePrime object created between both input notes
    """
    a_interval = interval.Interval(next_note.pitch - current_note.pitch)
    #To-fix: Fix intervals to be only perfect, major, or minor
    generic_interval = a_interval.generic.directed
    
    # Uses IOI if available. If not (because the note is at the end of the sequence), use its duration instead
    current_duration = current_note.interonset_interval if current_note.interonset_interval is not None else current_note.duration
    
    next_duration = next_note.interonset_interval if next_note.interonset_interval is not None else next_note.duration
    onset_ratio = next_duration / current_duration
    
    return SimpleNotePrime(generic_interval, onset_ratio, current_note.measure_number, current_note.note_measure_index)

    
def create_song_object(midi_filepath, song_name, song_index):
    """
    Creates a song object given a midi_filepath and some other identifying information
    
    :param str midi_filepath: the path to the midi file
    :param str song_name: the name of the song
    :param int song_index: the index of the song within its soundtrack
    :return: a song object for the given midi file
    """
    # Change from True to False for testing
    original_note_data = midi_to_notes_by_measure(midi_filepath, False)
    
    simple_notes_data: dict[str, list[list[SimpleNote]]] = {}
    sky_simple_notes_data: dict[str, list[SimpleNote]] = {}
    for part, measures in original_note_data.items():
        simple_notes_data[part], sky_simple_notes_data[part] = notes_by_measure_to_simple_notes(measures)
    
    prime_notes_data: dict[str, list[list[list[SimpleNotePrime]]]] = {}
    for part, simple_notes_list in simple_notes_data.items():
        prime_notes_data[part] = simple_notes_to_prime_notes(simple_notes_list)
    
    sky_prime_notes_data: dict[str, list[SimpleNotePrime]] = {}
    for part, sky_simple_notes in sky_simple_notes_data.items():
        sky_prime_notes_data[part] = simples_sky_notes_to_prime_sky_notes(sky_simple_notes)
    
    song = Song(song_index=song_index, song_name=song_name, original_notes_data=original_note_data,
                simple_notes_data=simple_notes_data, prime_notes_data=prime_notes_data,
                sky_simple_notes_data=sky_simple_notes_data, sky_prime_notes_data=sky_prime_notes_data)
    
    return song
    
def test_song():
    midi_filepath = "../../TestMidiFiles/Deltarune - My Castle Town.mid"
    song_name = "My Castle Town"
    song_index = 0
    
    song = create_song_object(midi_filepath, song_name, song_index)

    part = song.get_parts_list()[0]
    #21-24 and 29-31
    test1 = [x for x in song.sky_simple_notes_data[part] if 21 <= x.measure_number <= 24]
    test2 = [x for x in song.sky_prime_notes_data[part] if 21 <= x.measure_number <= 24]
    print_sky_notes(test1)
    print("")
    print_sky_notes(test2)
    

def note_to_string(general_note: GeneralNote):
    txt = ""
    if general_note.isNote:
        txt = "type: {0}, pitch: {1}, midi_pitch: {2}, duration_quarter: {3}, offset: {4}, velocity: {5}".format(
            "note", general_note.pitches[0].nameWithOctave, general_note.pitches[0].midi,
            general_note.quarterLength, general_note.offset, general_note.volume.velocity
        )
    elif general_note.isChord:
        txt = "type: {0}, pitches: {1}, midi_pitch: {2}, duration_quarter: {3}, offset: {4}, velocity: {5}".format(
            "chord", [p.nameWithOctave for p in general_note.pitches], [p.midi for p in general_note.pitches],
            general_note.quarterLength, general_note.offset, general_note.volume.velocity
        )
    elif general_note.isRest:
        txt = "type: {0}, duration_quarter: {1}, offset: {2}".format(
            "rest", general_note.quarterLength, general_note.offset
        )
        
    return txt
    
def print_midi_data(midi_data: dict[str, list[list[GeneralNote]]]):
    for track, measures in midi_data.items():
        print(track)
        for measure_number, measure in enumerate(measures):
            print("Measure: " + (measure_number + 1).__str__())
            for index, note in enumerate(measure):
                print(index.__str__() + " " + note_to_string(note))
                
def print_original_note_data(midi_data: dict[str, list[list[OriginalNote]]]):
    for track, measures in midi_data.items():
        print(track)
        for measure_number, measure in enumerate(measures):
            print("Measure: " + (measure_number + 1).__str__())
            for index, note in enumerate(measure):
                print(index.__str__() + " " + note.short_repr())


def print_sky_notes(sky_notes: list[SimpleNote] | list[SimpleNotePrime]):
    for index, sky_note in enumerate(sky_notes):
        print(index.__str__() + " " + sky_note.__str__())


def print_simple_notes(simple_notes: list[list[SimpleNote]]):
    for simple_note_chord in simple_notes:
        print(simple_note_chord.__repr__())


def print_prime_notes(prime_notes: list[list[list[SimpleNotePrime]]]):
    for chord_combinations in prime_notes:
        for note_combinations in chord_combinations:
            print(note_combinations.__repr__())
            

def print_midi_file():
    midi_filepath = "../../TestMidiFiles/Super Mario Galaxy - Rosalinas Comet Observatory 1 2 3.mid"
    original_note_data = midi_to_notes_by_measure(midi_filepath, True)
    #print_original_note_data(original_note_data)

    
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
    

if __name__ == "__main__":
    from src.motif_finder.tests import perform_all_tests
    # print_midi_file()
    perform_all_tests()
