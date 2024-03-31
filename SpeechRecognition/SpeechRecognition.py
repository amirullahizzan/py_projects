import deepspeech
import numpy as np
import pyaudio

# File paths for the DeepSpeech model and scorer
model_path = "c:/DEV/VSCode/Python Code/SpeechRecognition/deepspeech-0.9.3-models.pbmm"
lm_path = "c:/DEV/VSCode/Python Code/SpeechRecognition/deepspeech-0.9.3-models.scorer"

# Load the DeepSpeech model
model = deepspeech.Model(model_path)
# Enable the external scorer for better accuracy (if available)
if lm_path:
    lm = model.enableExternalScorer(lm_path)
print("Models loaded successfully.")

# Function to process audio input from the microphone
def process_audio_input():
    chunk = 1024  # Number of frames in a buffer
    sample_format = pyaudio.paInt16  # 16-bit resolution
    channels = 1  # Mono
    fs = 16000  # Sample rate (Hz)
    seconds = 5  # Duration of recording

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print('Recording...')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    # Store data in chunks for specified duration
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.int16))  # Convert audio data to numpy array

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print('Finished recording.')

    # Combine frames into a single numpy array
    audio_data = np.concatenate(frames)

    return audio_data, fs  # Return audio data and sample rate

# Function to perform speech recognition
def recognize_speech(audio_data):
    return model.stt(audio_data)

program_loop = True
# Process audio input and perform speech recognition
def main():
    while program_loop:
        audio_data, _ = process_audio_input()  # Ignore sample_rate for now
        recognized_text = recognize_speech(audio_data)
        print("Recognized text:", recognized_text)
        if recognized_text.lower() == "terminate" or recognized_text.lower() == "exit":
            break

if __name__ == "__main__":
    main()

print("Program Ends")