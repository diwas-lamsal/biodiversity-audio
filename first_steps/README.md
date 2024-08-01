Use the `create_repair_log.py` and `ffmpeg_repair_from_log.py` files to repair corrupt audio files. 

Given the root audio file directoy, `create_repair_log.py` notes all the corrupt audio files into a log as well as a pickle file. 

Given the log file path, the `ffmpeg_repair_from_log.py` will repair and replace the corrupt audio files. 