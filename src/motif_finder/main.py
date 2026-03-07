import dataclasses
from dataclasses import dataclass
import json
import os
from collections import Counter
from fractions import Fraction
from typing import Dict
from xml.sax import default_parser_list

import music21
from music21 import converter, note, midi, duration, meter, interval, pitch
from music21.interval import GenericInterval
from music21.meter import TimeSignature
from music21.midi import MidiFile
from music21.note import GeneralNote
from music21.stream import Measure, Score
from music21.stream.makeNotation import consolidateCompletedTuplets

from librosa import segment

import matplotlib.pyplot as plt
import matplotlib.colors

import numpy as np
from numba.typed.dictobject import new_dict
from scipy.signal import freqs


# To-Do
# - Create a test phrase group and supporting code

# To-Review
# - Review what objects are actually needed for the song class
# - Review if I should remove the SimpleNote class and replace it with GeneralNote from Music21

# To-Fix
# -

# Notes
# - .exec method for running string as code


@dataclass
class SimpleNote(object):
    pitch: int = 0
    duration: float = 0.0
    onset: float = 0.0
    interonset_interval: float | None = None  #Onset difference between this note and the next
    measure_number: int = 0
    note_measure_index: int = 0  #Index of the note within a measure (similar to offset)
    tie: str | None = None
    
    # def __eq__(self, other):
    #     return self.pitch == other.pitch and self.note_duration == other.note_duration and self.tie == other.tie
    
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
    original_notes_data: dict[str, list[list[GeneralNote]]]  #Full, original sequence grouped by measure
    simple_notes_data: dict[str, list[list[SimpleNote]]]  #All notes without rests, grouped by chord (not measure)
    prime_notes_data: dict[str, list[list[list[SimpleNotePrime]]]]  #All prime note combinations, grouped by chord
    sky_simple_notes_data: dict[str, list[SimpleNote]]  #Skyline notes without rests
    sky_prime_notes_data: dict[str, list[SimpleNotePrime]]  #Skyline prime notes without rests


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
    music_string_list: list[MusicPhrase]
    
    # Internal variables (don't need to be added with object instantiation)
    song_names: list[str]  #List of songs this phrase group appears in
    
    def __init__(self, music_string_list: list[MusicPhrase]):
        self.music_string_list = music_string_list
        self.song_names = []
    
    # Always use this method to add a phrase to a phrase_group object
    def add(self, music_string: MusicPhrase):
        self.music_string_list.append(music_string)
        
        if music_string.get_first_position().song_name not in self.song_names:
            song_name = music_string.get_first_position().song_name
            self.song_names.append(song_name)
    
    # Original phrase is defined as the first time the phrase appears
    def get_original_phrase(self):
        return self.music_string_list[0]
    
    
# Global Variables
song_dict: dict[str, Song] = {}
phrase_group_list: list[PhraseGroup] = []


def midi_to_measures(midi_path: str):
    # Extracts notes and rests from a MIDI file organized by track (Part).
    #
    # Parameters:
    #     midi_path (str): Path to the MIDI file.
    #
    # Returns:
    #     dict: A dictionary where each key is a track name (or index)
    #           and the value is a list of Measure objects.
    
    # Parse MIDI file
    score = converter.parse(midi_path,  quarterLengthDivisors=(2, 3, 4, 8))
    # score = score.explode()
    
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
            # print(part[music21.instrument.Instrument][0])
        
        if i == 0:
            time_signature = part[music21.meter.TimeSignature][0]
        else:
            part['Measure'][0].timeSignature = time_signature
            
        # Remake measures to account for new time signature
        part = part.makeMeasures()
        
        part = part.stripTies()
        part = part.makeTies()
        
        consolidateCompletedTuplets(part, onlyIfTied=False)
        
        # Check for repetitive parts with same part_name, and rename them (for example, piano sometimes has two parts)
        same_track_number = 1
        while part_name in midi_data:
            same_track_number += 1
            part_name = part_name + " " + same_track_number.__str__()
        
        # Gets list of measures (syntax is weird because of OrderedDictionary shenanigans)
        measure_list = []
        temp = list(part.measureOffsetMap().values())
        for measures_temp in temp:
            measure = measures_temp[0]
            
            measure_list.append(measure)
        midi_data[part_name] = measure_list

    return midi_data
    


def measure_to_notes(measure: Measure):
    # Flatten the stream to make sure we get all notes and rests
    flat_notes = measure.flatten().notesAndRests
    return flat_notes


def measures_to_notes_list(measures: list[Measure]):
    general_notes_list: list[list[GeneralNote]] = []
    for measure in measures:
        flat_notes = measure_to_notes(measure)
        general_notes_list.append(list(flat_notes))
    return general_notes_list


# Inputs measure and outputs both simple_notes and sky_simple_notes
# For both outputs, all rests are removed
# Measure number and offset is preserved by individual SimpleNote objects.
def measures_to_simple_notes(measures: list[Measure]):
    simple_notes: list[list[SimpleNote]] = []
    sky_simple_notes: list[SimpleNote] = []
    
    for measure in measures:
        # Within a measure, identity the melody by the highest note currently being played (skyline algorithm)
        # Keep offset of skyline notes to avoid overlap
        local_sky_offset = 0
        note_measure_index = 0
        for general_note in measure.notesAndRests:
            # Ignores all rests
            if not general_note.isRest:
                # Identify if general note is a tied note or not
                if general_note.tie is None:
                    tie_type = None
                else:
                    tie_type = general_note.tie.type
                
                # Get the highest pitch of chord or note
                # At the same time, make a SimpleNote object for each pitch in a chord
                pitch_list = [p.midi for p in general_note.pitches]
                highest_pitch = 0
                simple_note_chord: list[SimpleNote] = []
                
                sky_simple_note = None
                for pitch in pitch_list:
                    simple_note = SimpleNote(pitch=pitch, duration=general_note.quarterLength,
                                             onset=(measure.offset + general_note.offset), measure_number=general_note.measureNumber,
                                             note_measure_index=note_measure_index, tie=tie_type)
                    simple_note_chord.append(simple_note)
                    
                    if pitch > highest_pitch:
                        highest_pitch = pitch
                        sky_simple_note = simple_note
                
                # Append sky_simple_note if there is no overlap
                if general_note.offset >= local_sky_offset:
                    sky_simple_notes.append(sky_simple_note)
                    
                    # Adjust offset and then add duration
                    local_sky_offset = general_note.offset + general_note.quarterLength
                
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


def simples_sky_notes_to_prime_sky_notes(simple_sky_notes: list[SimpleNote]):
    prime_sky_notes: list[SimpleNotePrime] = []
    
    for index in range(len(simple_sky_notes) - 1):
        current_note = simple_sky_notes[index]
        next_note = simple_sky_notes[index + 1]
        prime_sky_notes.append(find_prime(current_note, next_note))
        
    return prime_sky_notes
        
def simple_notes_to_prime_notes(simple_notes: list[list[SimpleNote]]):
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
    
        
def find_prime(current_note: SimpleNote, next_note: SimpleNote):
    p1 = pitch.Pitch(current_note.pitch)
    p2 = pitch.Pitch(next_note.pitch)
    a_interval = interval.Interval(pitchStart=p1, pitchEnd=p2)
    generic_interval = a_interval.generic.directed
    
    # Uses IOI if available. If not (because the note is at the end of the sequence), use its duration instead
    current_duration = current_note.interonset_interval if current_note.interonset_interval is not None else current_note.duration
    next_duration = next_note.interonset_interval if next_note.interonset_interval is not None else next_note.duration
    onset_ratio = next_duration / current_duration
    
    return SimpleNotePrime(generic_interval, onset_ratio, current_note.measure_number, current_note.note_measure_index)


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
            if i+j < size:
                lag_matrix[i][j] = boolean_ssm[i+j][j]
            else:
                break
    return lag_matrix

#
# def traverse_boolean_ssm(boolean_ssm: list[list[bool]]):


def find_note_list_by_measure_range(simple_note_lists: list[list[SimpleNote]], measure_range: range):
    note_list: list[SimpleNote] = []
    for measure_number in measure_range:
        #TO-FIX: Reduce range so that it can't (somehow) be outside of range
        if measure_number >= len(simple_note_lists):
            continue
        
        note_list.extend(simple_note_lists[measure_number])
    return note_list
    

# Identify if a given leitmotif is in a song and where
# Note: Song is provided as a single list of SimpleNotePrime objects (analyzing regardless of measures)
# To-fix: Function doesn't take into account uneven chords
def query_leitmotif(query_phrase_group: PhraseGroup, current_song: Song):
    query = query_phrase_group.get_original_phrase().prime_notes
    found_query = False
    
    for part_name, simple_note_primes in current_song.prime_notes_data.items():
        note_part_index = 0
        
        while note_part_index < (len(simple_note_primes) - len(query)):
            query_index = 0
            
            # First checks if a chord contains the query's first note
            # After that, checks if the associated next note follows the query sequence
            follow_note = True
            next_note_index: int | None = None
            while query_index < len(query) and follow_note:
                follow_note = False
                chord_combinations = simple_note_primes[note_part_index + query_index]
                if next_note_index is None:
                    for note_combinations in chord_combinations:
                        for note_combination in note_combinations:
                            if query[query_index].compare_note(note_combination):
                                follow_note = True
                                next_note_index = note_combinations.index(note_combination)
                else:
                    for note_combination in chord_combinations[next_note_index]:
                        if query[query_index].compare_note(note_combination):
                            follow_note = True
                            next_note_index = chord_combinations[next_note_index].index(note_combination)
                
                query_index += 1
                
            # Found a match to the query
            if query_index == len(query):
                found_query = True
                
                # Create new PhrasePosition
                start_note = simple_note_primes[note_part_index][0][0]
                new_phrase_position = PhrasePosition(current_song.song_name, part_name, start_note.measure_number,
                                                     start_note.note_measure_index)
                
                # Update phrase group (unless new_phrase_position is already present)
                if new_phrase_position not in query_phrase_group.get_original_phrase().positions:
                    query_phrase_group.get_original_phrase().update(new_phrase_position)
                
                # Increment note index past successful query
                note_part_index += len(query)
            
            note_part_index += 1
            
    return found_query
            
            
def create_song_object(midi_filepath: str, song_name:str, song_index: int):
    midi_data = midi_to_measures(midi_filepath)
    
    original_note_data: dict[str, list[list[GeneralNote]]] = {}
    for part, measures in midi_data.items():
        original_note_data[part] = measures_to_notes_list(measures)
    
    simple_notes_data: dict[str, list[list[SimpleNote]]] = {}
    sky_simple_notes_data: dict[str, list[SimpleNote]] = {}
    for part, measures in midi_data.items():
        simple_notes_data[part], sky_simple_notes_data[part] = measures_to_simple_notes(measures)
    
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
            
# Functions for testing
def test_phrase_group():
    midi_filepath = "../../Hollow Knight Main Theme.mid"
    song_name = "Hollow Knight Main Theme"
    song_index = 0
    
    song = create_song_object(midi_filepath, song_name, song_index)
    
    # for part, simple_chords in song.simple_notes_data.items():
    #     print(part)
    #     for simple_chord in simple_chords:
    #         for simple_note in simple_chord:
    #             print(simple_note.__repr__())
    
    # Grabs first motif in Hollow Knight Main Theme
    query_measure = 0
    query_note_measure_index = 0
    query_size = 11-1
    
    # Searches through first part only
    first_part = next(iter(song.sky_prime_notes_data))
    start_note = song.sky_prime_notes_data[first_part][query_note_measure_index]
    motif_sequence = song.sky_prime_notes_data[first_part][query_note_measure_index:query_size]
    
    # Create relevant objects
    phrase_position = PhrasePosition(song.song_name, first_part, start_note.measure_number, start_note.note_measure_index)
    music_string = MusicPhrase(motif_sequence, 1, [phrase_position])
    phrase_group = PhraseGroup([music_string])
    
    query_leitmotif(phrase_group, song)
    
    print(phrase_group.__repr__())



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
    
def print_notes(midi_data: dict[str, list[Measure]]):
    for track, measures in midi_data.items():
        print(track)
        for measure_number, measure in enumerate(measures):
            print("Measure: " + (measure_number + 1).__str__())
            for index, note in enumerate(measure.notesAndRests):
                print(index.__str__() + " " + note_to_string(note))
                
def print_prime_notes(simple_midi_prime_data: Dict[str, list[list[SimpleNotePrime]]]):
    for track, note_prime_lists in simple_midi_prime_data.items():
        print(track)
        for measure_number, note_prime_list in enumerate(note_prime_lists):
            print("Measure: " + measure_number.__str__())
            for index, note_prime in enumerate(note_prime_list):
                print(index.__str__() + " " + note_prime.__str__())
                
                
def plot_colored_grid(data, song_name, part_name):
    # Color for False and True
    cmap = matplotlib.colors.ListedColormap(['black', 'white'])
    
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 200
    plt.imshow(data, cmap=cmap)
    plt.title("{0}: {1}".format(song_name, part_name))
    plt.show()
    
    
    
def process_midi_file(midi_filepath: str, song_name:str, song_index: int):
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
    
    #return simple_midi_data, lag_matrix_list

def print_midi_file():
    midi_filepath = "../../Hollow Knight Main Theme.mid"
    midi_data = midi_to_measures(midi_filepath)
    print_notes(midi_data)


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
     
     
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
        
def process_phrase_dictionary():
    # Create phrase Dictionary
    #phrase_dictionary = create_phrase_dictionary()

    # Setup up code to read each midi file for Super Mario Galaxy
    folder_name = "MidiFiles/superMarioGalaxy"
    game_name = "Super Mario Galaxy"

    midi_files = []
    for (dirpath, dirnames, filenames) in os.walk(folder_name):
        midi_files.extend(filenames)

    for temp_song_index, midi_file in enumerate(midi_files):

        song_name = midi_file
        simple_midi_data, lag_matrix_list = process_midi_file(folder_name + "/" + midi_file)

        # values = list(simple_midi_data.values())
        # for value in values:
        #     print(len(value))

        # add_to_phrase_dictionary(phrase_dictionary, temp_song_index, song_name, lag_matrix_list, simple_midi_data)

    # json_str = json.dumps(phrase_dictionary, indent=4, cls=EnhancedJSONEncoder)
    # with open("SuperMarioGalaxy.json", "w") as f:
    #     f.write(json_str)
    
    

if __name__ == "__main__":
    # Separate Tests
    # print_midi_file()
    
    test_phrase_group()
    
    # test_single_file()
    # test_multiple_files()
    
    # process_phrase_dictionary()

    # Print structured summary
    # print_prime_notes(simple_midi_prime_data)
