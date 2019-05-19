########################################
	10-707 PROJECT README
########################################

Step 1: Extract images and MFCC features using the file: joint_feature_extractor.py
Step 2: Create the test, train and val sets using file: create_train_test_and_val_sets.py
Step 3: Now, we need a way to match the number of audio to the number of video files, this is a tedious process, use file: expand_audio_files.py

Step 4: Select a model for training, validation and test


++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TO RUN THE CODE USE:

python main.py <path_to_audio_dataset> <path_to_video_dataset> <temporary_directory_path> <model_selection>

	- Audio dataset includes both speech and song data
	- Video dataset includes only speech videos
	- Temporary directory does all pre-processing and stores the data (THIS MAY TAKE OVER 15 minutes to run)
	- Type any of the following for model selection
		- only_audio
		- only_video
		- joint_cat
		- joint_mlp
		- probe
		- joint_loss

EXAMPLE:

python main.py ~/audio_dataset ~/video_dataset ~/tmpdir probe