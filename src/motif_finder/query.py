from main import SimpleNote
from main import SimpleNotePrime
from main import PhraseGroup
from main import PhrasePosition
from main import Song
import numpy as np

# Uses the Smith-Waterman algorithm on two SimpleNotePrime lists
# Returns a similarity value as a percentage of the resulting score and the max possible score
# Currently only measures pitch
def smith_waterman_compare(query: list[SimpleNotePrime], sequence: list[SimpleNotePrime]):
	# To-do: Add test cases that prematurely end compare
	
	w1 = 2  #linear gap penalty
	
	# Add an empty object to beginning of the list for easier comparisons
	empty = SimpleNotePrime()
	query.insert(0, empty)
	sequence.insert(0, empty)
	
	scoring_matrix = np.zeros((len(query), len(sequence)))
	max_value: int = 0
	max_value_position: tuple[int, int] = (0, 0)
	
	# create another matrix to store which cell's value came from (mirrors the scoring matrix)
	traceback_matrix = np.empty((len(query), len(sequence)))

	# Populate scoring matrix
	for query_index in range(1, scoring_matrix.shape[0]):
		for sequence_index in range(1, scoring_matrix.shape[1]):
			match = find_sub_matrix_value(query[query_index].generic_interval, sequence[sequence_index].generic_interval)
			delete = scoring_matrix[query_index - 1, sequence_index] - w1
			insert = scoring_matrix[query_index, sequence_index - 1] - w1
			cell_value = max(match, delete, insert, 0)
			scoring_matrix[query_index, sequence_index] = cell_value
			
			# Update maximum value for easier traceback (repeat values override to get last occurring maximum)
			if cell_value >= max_value:
				max_value = cell_value
				max_value_position = (query_index, sequence_index)
				
			# Update traceback matrix
			if cell_value == match:
				traceback_matrix[query_index, sequence_index] = (query_index - 1, sequence_index - 1)
			elif cell_value == delete:
				traceback_matrix[query_index, sequence_index] = (query_index - 1, sequence_index)
			elif cell_value == insert:
				traceback_matrix[query_index, sequence_index] = (query_index, sequence_index - 1)
			# If cell_value is 0, leave it empty
				
	# Retrace path
	i, j = max_value_position
	while scoring_matrix[i, j] != 0:
		i, j = traceback_matrix[i, j]
			
	
def find_sub_matrix_value(interval_1: int, interval_2: int):
	value = 3 - 2*abs(interval_1 - interval_2)
	if value < -3:
		value = -3
	return value


# Should have a "tolerance level" for how much a phrase can deviate before quitting the search
# [Under Construction]
def query_similar_leitmotif(query_phrase_group: PhraseGroup, current_song: Song):
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