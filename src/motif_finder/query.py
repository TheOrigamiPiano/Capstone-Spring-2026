from main import SimpleNote, MusicPhrase
from main import SimpleNotePrime
from main import PhraseGroup
from main import PhrasePosition
from main import Song
import numpy as np

similarity_threshold = 0.75
# Each test sequence will be 1.5x bigger than the query for efficiency.
# Similarity ratio will still be based on the query length
phrase_length_difference = 1.5

# Uses the Smith-Waterman algorithm on two SimpleNotePrime lists
# Returns a similarity value as a percentage of the resulting score and the max possible score, as well as both of the
# identified matching phrases
# Currently only measures pitch
def smith_waterman_compare(query: list[SimpleNotePrime], sequence: list[SimpleNotePrime]):
	# To-do: Add test cases that prematurely end compare
	
	w1 = 2  #linear gap penalty
	
	# Add an empty object to beginning of the list for easier comparisons
	empty = SimpleNotePrime()
	new_query = query.copy()
	new_query.insert(0, empty)
	new_sequence = sequence.copy()
	new_sequence.insert(0, empty)
	
	scoring_matrix = np.zeros((len(new_query), len(new_sequence)))
	max_value: int = 0
	max_value_position: tuple[int, int] = (0, 0)
	
	# create another matrix to store which cell's value came from (mirrors the scoring matrix)
	traceback_matrix = np.full((len(new_query), len(new_sequence)), (0, 0), dtype=(int, 2))

	# Populate scoring matrix
	for query_index in range(1, scoring_matrix.shape[0]):
		for sequence_index in range(1, scoring_matrix.shape[1]):
			match = (scoring_matrix[query_index - 1, sequence_index - 1] +
					 find_sub_matrix_value(new_query[query_index].generic_interval, new_sequence[sequence_index].generic_interval))
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
				
				
	# Retrace path
	i, j = max_value_position
	last_sequence_position = j
	query_pattern = []
	sequence_pattern = []
	while scoring_matrix[i, j] != 0:
		query_pattern.insert(0, new_query[i])
		sequence_pattern.insert(0, new_sequence[j])
		i, j = traceback_matrix[i, j]
		
	# print(scoring_matrix.__repr__())
	# print(traceback_matrix.__repr__())

	# Return ratio of max_value to the highest possible score (3 * the maximum length of most similar segments)
	similarity = max_value / (3 * (len(new_query) - 1))
	last_measure = new_sequence[last_sequence_position].measure_number
	return similarity, sequence_pattern, last_measure
	
	
def find_sub_matrix_value(interval_1: int, interval_2: int):
	value = 3 - 2*abs(interval_1 - interval_2)
	if value < -3:
		value = -3
	return value


# Should have a "tolerance level" for how much a phrase can deviate before quitting the search
def query_similar_skyline_leitmotif(query_phrase_group: PhraseGroup, current_song: Song):
	query = query_phrase_group.get_original_phrase().prime_notes
	found_query = False
	sequence_length = int(len(query)*phrase_length_difference)  #int() to round down
	
	for part_name, sky_prime_notes in current_song.sky_prime_notes_data.items():
		note_part_index = 0
		song_iter = iter(sky_prime_notes)
		next(song_iter)  #Start iterator on first note
		current_measure = 1
		
		while note_part_index < (len(sky_prime_notes) - sequence_length):
			sequence = sky_prime_notes[note_part_index:(note_part_index + sequence_length)]
			# print(query.__repr__())
			# print(sequence.__repr__())
			# print(current_song.sky_simple_notes_data[part_name][note_part_index:(note_part_index + sequence_length)].__repr__())
			similarity, sequence_pattern, last_measure = smith_waterman_compare(query, sequence)
			# print(similarity)
			
			# Found a match to the query
			if similarity > similarity_threshold:
				found_query = True
				
				# Create new PhrasePosition
				start_note = sky_prime_notes[note_part_index]
				new_phrase_position = PhrasePosition(current_song.song_name, part_name, start_note.measure_number,
													 start_note.note_measure_index)
				
				phrase_updated = False
				for phrase in query_phrase_group.music_phrase_list:
					#Quit if this phrase was already present
					if new_phrase_position in phrase.positions:
						phrase_updated = True
						break
					#Update phrase if it already exists
					elif phrase.prime_notes == sequence_pattern:
						phrase_updated = True
						phrase.update(new_phrase_position)
					
				#If phrase is not already in phrase_group, create a new MusicPhrase object
				if not phrase_updated:
					new_music_phrase = MusicPhrase(sequence_pattern, 1, [new_phrase_position])
					query_phrase_group.add(new_music_phrase)
				
				# Increment note index to the start of the last measure found in sequence_pattern
				current_measure = last_measure
			
			# Otherwise, increment note index to the start of the next measure
			else:
				current_measure += 1
			
			while next(song_iter).measure_number < current_measure:
				note_part_index += 1
			note_part_index += 1
			
			#To-fix: Because of the skip_length, ensure one final check to ensure all
			#data in the piece has been looked at
	
	return found_query




