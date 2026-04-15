from typing import Dict

import numpy as np
from librosa import segment
from music21 import interval, pitch
from music21.note import GeneralNote

# Iterates through midi_data, and combines notes into chords regardless of velocity
def ignore_velocity(midi_data):
	# Iteratively stores notes with the same offset value, along with the notes index.
	# Resets when finding a note with a different offset
	notes_with_same_offset: Dict[GeneralNote, int] = {}
	
	for track, general_notes in midi_data.items():
		for index, current_note in enumerate(general_notes):
			current_note: GeneralNote  # Type hint for GeneralNote
			
			# Skips all rests
			if current_note.isRest:
				continue
			
			# Near beginning of iteration, add the first note to list and skip rest of code (nothing to check yet)
			if not notes_with_same_offset:
				notes_with_same_offset[current_note] = index
				continue
			
			# When finding a note with new offset, reset list to that note
			# Else if offset is the same, then check if they can be combined into a chord.
			#   If so, then update chord into midi_data. Otherwise, add note to the list
			for previous_note in notes_with_same_offset.keys():
				if current_note.offset != previous_note.offset:
					notes_with_same_offset = {current_note: index}
				else:
					if previous_note.quarterLength == current_note.quarterLength:
						# Add current pitches to previous note
						for pitch_group in current_note.pitches:
							pitch_group: tuple
							for pitch in pitch_group:
								previous_note.pitches.__add__(pitch)
						
						# Replace previous note in midi_data
						previous_index = notes_with_same_offset[previous_note]
						midi_data[track][previous_index] = previous_note
						
						# Delete current note in midi_data
						del midi_data[track][index]
	
	return midi_data


def compare_pitch_names(p1: str, p2: str):
	a_interval = interval.Interval(pitchStart=pitch.Pitch(p1), pitchEnd=pitch.Pitch(p2))
	return a_interval.semitones

# Convert diagonals into rows. Return numpy array
def create_lag_matrix(boolean_ssm: list[list[bool]]):
	y = np.array([np.array(xi) for xi in boolean_ssm])
	lag_matrix = segment.recurrence_to_lag(y, pad=False)
	return lag_matrix


# Adds an entire song (including its tracks) to the phrase dictionary
def add_to_phrase_dictionary(phrase_dictionary: Dict[int, PhraseGroup], song_index: int,
							 song_name: str, lag_matrix_list: list[list[list[bool]]],
							 simple_midi_data: Dict[str, list[list[SimpleNote]]]):
	index = 0
	for track, simple_note_lists in simple_midi_data.items():
		
		lag_matrix = lag_matrix_list[index]
		note_list_length = len(simple_note_lists)
		print(song_name + " / " + track)
		
		# print("SNL: " + len(simple_note_lists).__str__())
		
		for row_index, lag_row in enumerate(lag_matrix):
			print("Row: " + row_index.__str__())
			# Skip first row
			if row_index == 0:
				continue
			
			# Skip rows after list length
			elif row_index >= note_list_length:
				continue
			
			# Variables to find repeated phrase
			found_phrase: bool = False  # A new phrase is found and being measured between cells
			starting_measure_1: int = 0  # First starting measure of the phrase
			starting_measure_2: int = 0  # Second starting measure of the phrase
			phrase_length: int = 0  # Length in measures
			for column_index, cell in enumerate(lag_row):
				# print("Column: " + column_index.__str__())
				
				# Skip columns after list length
				if column_index >= note_list_length:
					continue
				
				if cell:
					if not found_phrase:
						found_phrase = True
						starting_measure_1 = column_index
						starting_measure_2 = column_index + row_index
						phrase_length = 0
					phrase_length += 1
				
				# End current phrase and create MusicString for it.
				# - If the MusicString is new, then add a new entry to the dictionary
				# - Else, update existing motif group accordingly
				elif found_phrase:
					found_phrase = False
					
					# PhrasePosition objects
					first_position = PhrasePosition(song_index, song_name, track, starting_measure_1, 0)
					second_position = PhrasePosition(song_index, song_name, track, starting_measure_2, 0)
					
					# Search for pre-existing match in phrase dictionary
					phrase_match_found = False
					
					for int_key, phrase_group in phrase_dictionary.items():
						phrase_key = phrase_group.phrase_key
						music_string_list = phrase_group.music_phrase_list
						
						# Look for matches within the same song
						if phrase_key.song_names.__contains__(song_name):
							
							# Look for matches within the same track
							if phrase_key.first_position.track_name == track:
								
								exact_match = False
								similar_match = False
								chosen_music_string: MusicString | None = None
								note_list: list[SimpleNote] = []
								chosen_position: PhrasePosition | None = None
								
								for music_string in music_string_list:
									
									# Find a match (either exact or similar)
									match_first_position = music_string.positions.__contains__(first_position)
									match_second_position = music_string.positions.__contains__(second_position)
									
									# Ignore if both positions are already in music_string
									if match_first_position and match_second_position:
										continue
									
									# If only one position is accounted for, then process the other
									# (Steps change based on whether the match is exact or similar)
									elif match_first_position or match_second_position:
										
										# Recreate MusicString for each position to determine exact or similar
										if match_first_position:
											range_2 = range(starting_measure_2, starting_measure_2 + phrase_length)
											note_list = find_note_list_by_measure_range(simple_note_lists, range_2)
											chosen_position = second_position
										else:
											range_1 = range(starting_measure_1, starting_measure_1 + phrase_length)
											note_list = find_note_list_by_measure_range(simple_note_lists, range_1)
											chosen_position = first_position
										
										# print(note_list)
										# print("")
										if music_string.note_list.__eq__(note_list):
											chosen_music_string = music_string
											exact_match = True
											
											# If an exact match is found, then no need to search further
											break
										else:
											similar_match = True
								
								# If there is an exact match, update positions list and frequency
								if exact_match:
									# Update chosen position
									chosen_music_string.positions.append(chosen_position)
									
									# Update frequency
									chosen_music_string.frequency += 1
								
								# If there is a similar match, create new MusicString in phrase group
								elif similar_match:
									new_music_string = MusicString(note_list, 1, [chosen_position])
									music_string_list.append(new_music_string)
							
							# Currently, ignore potential matches within other tracks
							else:
								continue
					
					# If no matches are found, then create new phrase group
					if not phrase_match_found:
						
						# Create phrase_key
						phrase_key = PhraseKey(first_position, [song_name])
						
						# Create MusicString objects
						# - One if the phrases are exact, and two if they are similar
						range_1 = range(starting_measure_1, starting_measure_1 + phrase_length)
						first_note_list = find_note_list_by_measure_range(simple_note_lists, range_1)
						
						range_2 = range(starting_measure_2, starting_measure_2 + phrase_length)
						second_note_list = find_note_list_by_measure_range(simple_note_lists, range_2)
						
						new_music_string_list: list[MusicString] = []
						
						if first_note_list.__eq__(second_note_list):
							new_music_string = MusicString(first_note_list, 2, [first_position, second_position])
							new_music_string_list.append(new_music_string)
						else:
							first_music_string = MusicString(first_note_list, 1, [first_position])
							second_music_string = MusicString(second_note_list, 1, [second_position])
							new_music_string_list.append(first_music_string)
							new_music_string_list.append(second_music_string)
						
						# Add new entry to dictionary
						new_phrase_group = PhraseGroup(phrase_key, new_music_string_list)
						new_int_key = len(list(phrase_dictionary.keys()))
						
						phrase_dictionary[new_int_key] = new_phrase_group


# Assumes exactness was already checked for
def check_insertion(query_note: SimpleNotePrime, current_chord_combinations: list[list[SimpleNotePrime]],
					next_chord_combinations: list[list[SimpleNotePrime]]):
	# Pitch
	possible_insertion_paths = list[(int, int)]
	for note_chord_index, current_note_combinations in enumerate(current_chord_combinations):
		for next_note_chord_index, current_note_combination in enumerate(current_note_combinations):
			interval = current_note_combination.generic_interval
			for next_note_combination in next_chord_combinations[next_note_chord_index]:
				interval += next_note_combination.generic_interval

# def check_deletion():
#
# def check_translation():
#
# def check_exact():

# # Only returns similarity value
# def smith_waterman_compare_simple(query: list[SimpleNotePrime], sequence: list[SimpleNotePrime]):
# 	w1 = 2  # linear gap penalty
#
# 	# Add an empty object to beginning of the list for easier comparisons
# 	empty = SimpleNotePrime()
# 	query.insert(0, empty)
# 	sequence.insert(0, empty)
#
# 	scoring_matrix = np.zeros((len(query), len(sequence)))
# 	max_value: int = 0
# 	max_value_position: tuple[int, int] = (0, 0)
#
# 	# Populate scoring matrix
# 	for query_index in range(1, scoring_matrix.shape[0]):
# 		for sequence_index in range(1, scoring_matrix.shape[1]):
# 			match = find_sub_matrix_value(query[query_index].generic_interval,
# 										  sequence[sequence_index].generic_interval)
# 			delete = scoring_matrix[query_index - 1, sequence_index] - w1
# 			insert = scoring_matrix[query_index, sequence_index - 1] - w1
# 			cell_value = max(match, delete, insert, 0)
# 			scoring_matrix[query_index, sequence_index] = cell_value
#
# 			# Update maximum value for easier traceback (repeat values override to get last occurring maximum)
# 			if cell_value >= max_value:
# 				max_value = cell_value
# 				max_value_position = (query_index, sequence_index)
#
# 	# Return ratio of max_value to the highest possible score (3 * the maximum length of most similar segments)
# 	return max_value / (3 * max(max_value_position))

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