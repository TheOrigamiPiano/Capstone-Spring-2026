from query import query_similar_skyline_leitmotif, smith_waterman_compare
from main import PhraseGroup, SimpleNotePrime
from main import PhrasePosition
from main import Song
from main import SimpleNote, MusicPhrase, create_song_object

def perform_all_tests():
	#Self-compare tests
	# self_compare_tests()
	
	# Cross-compare tests
	cross_compare_tests()
	
	# Other
	# temp_test()
	
	
def perform_and_print_self_test(song: Song, part: str, measure_number: int, measure_note_index: int, query_size: int):
	phrase_group = retrieve_phrase(song, part, measure_number, measure_note_index, query_size)
	query_similar_skyline_leitmotif(phrase_group, song)
	
	for music_phrase in phrase_group.music_phrase_list:
		# print(str(len(music_phrase.prime_notes)) + " " + music_phrase.__repr__())
		# print(song.simple_notes_data)
		print(str(len(music_phrase.prime_notes)) + " " + music_phrase.positions.__repr__())


def perform_and_print_cross_test(query_song: Song, part: str, measure_number: int, measure_note_index: int,
								 query_size: int, target_song: Song):
	query_phrase_group = retrieve_phrase(query_song, part, measure_number, measure_note_index, query_size)
	query_similar_skyline_leitmotif(query_phrase_group, target_song)
	
	for music_phrase in query_phrase_group.music_phrase_list:
		# print(str(len(music_phrase.prime_notes)) + " " + music_phrase.__repr__())
		print(str(len(music_phrase.prime_notes)) + " " + music_phrase.positions.__repr__())


def self_compare_tests():
	# 1. My Castle Town (Deltarune)
	# 1.1: Melody 1
	print("1. My Castle Town (Deltarune)")
	print("1.1")
	midi_filepath = "../../TestMidiFiles/Deltarune - My Castle Town.mid"
	song_name = "My Castle Town"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[0]
	
	perform_and_print_self_test(song, part, 1, 0, 9 - 1)
	
	# 1.2: Melody 2
	print("\n1.2")
	perform_and_print_self_test(song, part, 17, 0, 8 - 1)
	
	# 2. Dark Sanctuary (Deltarune)
	print("\n2. Dark Sanctuary (Deltarune)")
	midi_filepath = "../../TestMidiFiles/Deltarune - Dark Sanctuary (v2).mid"
	song_name = "Dark Sanctuary"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[0]
	
	perform_and_print_self_test(song, part, 17, 0, 11 - 1)
	# Only finding the first half of the melody, so it might be able to find the similar patterns later
	
	# 3. Hollow Knight Main Theme
	print("\n3. Hollow Knight Main Theme")
	midi_filepath = "../../TestMidiFiles/Hollow Knight Main Theme.mid"
	song_name = "Hollow Knight Main Theme"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[0]
	
	perform_and_print_self_test(song, part, 1, 0, 11 - 1)
	
	# 4. Rosalina's Comet Observatory (Super Mario Galaxy)
	print("\n4. Rosalina's Comet Observatory (Super Mario Galaxy)")
	midi_filepath = "../../TestMidiFiles/Super Mario Galaxy - Rosalinas Comet Observatory 1 2 3.mid"
	song_name = "Rosalina's Comet Observatory"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[0]
	
	perform_and_print_self_test(song, part, 9, 0, 11 - 1)
	# Note, the main melody is a longer phrase, but I'm just finding the first quarter of it for simplicity.
	
	# 5. Bowser Choir (Super Mario Galaxy)
	print("\n5. Bowser Choir (Super Mario Galaxy)")
	midi_filepath = "../../TestMidiFiles/Super_Mario_Galaxy_Bowser_Choir_v3.mid"
	song_name = "Bowser Choir"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[1]
	
	perform_and_print_self_test(song, part, 11, 0, 21 - 1)
	
	# 6. Toriel's Theme (Undertale)
	print("\n6. Toriel's Theme (Undertale)")
	midi_filepath = "../../TestMidiFiles/Undertale-Toriels_Theme_v2.mid"
	song_name = "Toriel's Theme"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[0]
	
	perform_and_print_self_test(song, part, 1, 0, 44 - 1)
	
	# 7. White Palace (Hollow Knight)
	print("\n7. White Palace (Hollow Knight)")
	midi_filepath = "../../TestMidiFiles/White_Palace_Hollow_Knight_v2.mid"
	song_name = "White Palace"
	song = create_song_object(midi_filepath, song_name, 0)
	part = song.get_parts_list()[1]
	
	perform_and_print_self_test(song, part, 17, 0, 10 - 1)
	
	
def cross_compare_tests():
	# 1. My Castle Town in Dark Sanctuary (Deltarune)
	print("\n1. My Castle Town in Dark Sanctuary (Deltarune)")
	query_song_name = "My Castle Town"
	query_song = create_song_object("../../TestMidiFiles/Deltarune - My Castle Town.mid", query_song_name, 0)
	part = query_song.get_parts_list()[0]
	
	target_song_name = "Dark Sanctuary"
	target_song = create_song_object("../../TestMidiFiles/Deltarune - Dark Sanctuary (v2).mid", target_song_name, 1)
	
	perform_and_print_cross_test(query_song, part, 17, 0, 9 - 1, target_song)
	
	# 2. Rosalina's Comet Observatory in Family (Super Mario Galaxy)
	print("\n2. Rosalina's Comet Observatory in Family (Super Mario Galaxy)")
	query_song_name = "Rosalina's Comet Observatory"
	query_song = create_song_object("../../TestMidiFiles/Super Mario Galaxy - Rosalinas Comet Observatory 1 2 3.mid", query_song_name, 0)
	part = query_song.get_parts_list()[0]
	
	target_song_name = "Family"
	target_song = create_song_object("../../TestMidiFiles/Super Mario Galaxy - Family.mid", target_song_name, 1)
	
	perform_and_print_cross_test(query_song, part, 9, 0, 11 - 1, target_song)
	
	# 3. White Palace in Resting Grounds (Hollow Knight)
	print("\n3. White Palace in Resting Grounds (Hollow Knight)")
	query_song_name = "White Palace"
	query_song = create_song_object("../../TestMidiFiles/White_Palace_Hollow_Knight_v2.mid",
									query_song_name, 0)
	part = query_song.get_parts_list()[1]
	
	target_song_name = "Resting Grounds"
	target_song = create_song_object("../../TestMidiFiles/Hollow Knight - Resting Grounds.mid", target_song_name, 1)
	
	perform_and_print_cross_test(query_song, part, 17, 0, 4 - 1, target_song)
	

def temp_test():
	query_song_name = "My Castle Town"
	query_song = create_song_object("../../TestMidiFiles/Deltarune - My Castle Town.mid", query_song_name, 0)
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
	
	similarity, sequence_pattern, start_note_position, last_measure = smith_waterman_compare(
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
	
	# print(song.sky_simple_notes_data[part][start_index:start_index + query_size + 1])
	
	# Create relevant objects
	phrase_position = PhrasePosition(song.song_name, part, start_note.measure_number,
									 start_note.note_measure_index)
	music_string = MusicPhrase(motif_sequence, 1, [phrase_position])
	phrase_group = PhraseGroup([music_string])
	
	return phrase_group
	
	