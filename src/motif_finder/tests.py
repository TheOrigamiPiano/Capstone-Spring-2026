from query import query_similar_skyline_leitmotif, smith_waterman_compare
from main import PhraseGroup, SimpleNotePrime
from main import PhrasePosition
from main import Song
from main import SimpleNote, MusicPhrase, create_song_object

def perform_all_tests():
	#Self-compare tests
	self_compare_tests()
	
	# Cross-compare tests
	cross_compare_tests()
	
	# Other
	temp_test()
	
	
def perform_and_print_self_test(song: Song, part: str, measure_number: int, measure_note_index: int, query_size: int):
	phrase_group = retrieve_phrase(song, part, measure_number, measure_note_index, query_size)
	query_similar_skyline_leitmotif(phrase_group, song)
	
	for music_phrase in phrase_group.music_phrase_list:
		print(str(len(music_phrase.prime_notes)) + " " + music_phrase.__repr__())


def perform_and_print_cross_test(query_song: Song, part: str, measure_number: int, measure_note_index: int,
								 query_size: int, target_song: Song):
	query_phrase_group = retrieve_phrase(query_song, part, measure_number, measure_note_index, query_size)
	query_similar_skyline_leitmotif(query_phrase_group, target_song)
	
	for music_phrase in query_phrase_group.music_phrase_list:
		print(str(len(music_phrase.prime_notes)) + " " + music_phrase.__repr__())


def self_compare_tests():
	# 1. My Castle Town (Deltarune)
	# 1.1: Melody 1
	print("1. My Castle Town (Deltarune)")
	print("1.1")
	midi_filepath = "../../Deltarune - My Castle Town.mid"
	song_name = "My Castle Town"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[0]
	
	perform_and_print_self_test(song, part, 0, 0, 9 - 1)
	
	# 1.2: Melody 2
	print("\n1.2")
	
	perform_and_print_self_test(song, part, 17, 0, 9 - 1)
	
	# 2. Hollow Knight Main Theme
	print("\n2. Hollow Knight Main Theme")
	midi_filepath = "../../Hollow Knight Main Theme.mid"
	song_name = "Hollow Knight Main Theme"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[0]
	
	perform_and_print_self_test(song, part, 0, 0, 11 - 1)
	
	
def cross_compare_tests():
	# 1. My Castle Town in Dark Sanctuary (Deltarune)
	print("\n1. My Castle Town in Dark Sanctuary (Deltarune)")
	query_song_name = "My Castle Town"
	query_song = create_song_object("../../Deltarune - My Castle Town.mid", query_song_name, 0)
	part = query_song.get_parts_list()[0]
	
	target_song_name = "Dark Sanctuary"
	target_song = create_song_object("../../Deltarune - Dark Sanctuary (v2).mid", target_song_name, 1)
	
	# target_song.print_simple_notes_data()
	
	perform_and_print_cross_test(query_song, part, 17, 0, 9 - 1, target_song)
	

def temp_test():
	query_song_name = "My Castle Town"
	query_song = create_song_object("../../Deltarune - My Castle Town.mid", query_song_name, 0)
	part = next(iter(query_song.sky_prime_notes_data))
	query_phrase_group = retrieve_phrase(query_song, part, 17, 0, 9 - 1)
	
	test_sequence = []
	test_sequence.append(SimpleNotePrime(3, 0.5, 49, 0))
	test_sequence.append(SimpleNotePrime(3, 1, 49, 1))
	test_sequence.append(SimpleNotePrime(6, 3, 49, 2))
	test_sequence.append(SimpleNotePrime(-2, 0.333, 50, 0))
	test_sequence.append(SimpleNotePrime(-3, 2, 50, 1))
	test_sequence.append(SimpleNotePrime(-4, 1, 51, 0))
	test_sequence.append(SimpleNotePrime(2, 1, 51, 1))
	test_sequence.append(SimpleNotePrime(3, 1, 52, 0))
	test_sequence.append(SimpleNotePrime(2, 4, 52, 1))
	
	similarity, sequence_pattern, last_measure = smith_waterman_compare(
		query_phrase_group.get_original_phrase().prime_notes, test_sequence)
	print(similarity)


# Finds and retrieves a query specified by part, measure number, measure_note_index, and size
# Note: Phrase must be found within the skyline part
def retrieve_phrase(song: Song, part: str, measure: int, measure_note_index: int, query_size: int):
	
	#Find initial note index from measure number
	initial_index = 0
	song_iter = iter(song.sky_prime_notes_data[part])
	while next(song_iter).measure_number < measure:
		initial_index += 1
	
	start_index = initial_index + measure_note_index
	start_note = song.sky_prime_notes_data[part][start_index]
	motif_sequence = song.sky_prime_notes_data[part][start_index:start_index + query_size]
	
	# Create relevant objects
	phrase_position = PhrasePosition(song.song_name, part, start_note.measure_number,
									 start_note.note_measure_index)
	music_string = MusicPhrase(motif_sequence, 1, [phrase_position])
	phrase_group = PhraseGroup([music_string])
	
	return phrase_group
	
	