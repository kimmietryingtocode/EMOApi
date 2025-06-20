import os
import wave
import struct

# Create samples directory if it doesn't exist
os.makedirs('samples', exist_ok=True)

# Parameters for a 3-second silent WAV
n_channels = 1
sampwidth = 2  # 2 bytes per sample (16-bit)
framerate = 44100  # samples per second
duration_seconds = 3
n_frames = framerate * duration_seconds

# Generate silent frames
silent_frames = [0] * n_frames

# Write the WAV file
with wave.open('samples/3s_clip.wav', 'w') as wf:
    wf.setnchannels(n_channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    # Pack frames as 16-bit little-endian
    frames_bytes = b''.join(struct.pack('<h', frame) for frame in silent_frames)
    wf.writeframes(frames_bytes)

print("Created 'samples/3s_clip.wav' (3 seconds of silence)")
