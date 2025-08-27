import whisper, pyaudio, numpy as np, queue, threading, time, sys

CHUNK_DURATION = 4
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16

model = whisper.load_model("turbo")  # consider "small"/"base" for lower latency
audio_queue = queue.Queue()

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=1024)

stop_event = threading.Event()

def record_audio():
    print("Recording... Press Ctrl+C to stop.", flush=True)
    try:
        while not stop_event.is_set():
            frames = []
            for _ in range(0, int(SAMPLE_RATE / 1024 * CHUNK_DURATION)):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.int16))
            audio_data = np.concatenate(frames).astype(np.float32) / 32768.0
            audio_queue.put(audio_data)
    except Exception as e:
        print(f"record_audio error: {e}", flush=True)
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass

def transcribe_loop():
    CONF_THRESHOLD = -1.0
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        print("Transcribing...", flush=True)
        traffic_prompt = "The driver is describing the current traffic condition. " \
                 "Possible words include: heavy traffic, light traffic, accident, congestion, pedestrians, vehicles."

        result = model.transcribe(
            audio_data, language='en', fp16=False, temperature=0.0,
            compression_ratio_threshold=2.4, logprob_threshold=-1.0,
            no_speech_threshold=0.6, prompt=traffic_prompt
        )
        segments = result.get("segments", [])
        if not segments:
            continue
        min_conf = min(s.get("avg_logprob", -999) for s in segments)
        if min_conf < CONF_THRESHOLD:
            print(f"Low confidence ({min_conf:.2f}) — skipping noisy chunk", flush=True)
            continue
        print("Transcription:", result['text'].strip(), flush=True)

if __name__ == "__main__":
    try:
        record_thread = threading.Thread(target=record_audio, daemon=True)
        transcribe_thread = threading.Thread(target=transcribe_loop, daemon=True)
        record_thread.start()
        transcribe_thread.start()

        # main thread is free now — do other work or just sleep
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        # don’t join forever; give short grace if you want
        for t in (record_thread, transcribe_thread):
            if t.is_alive():
                t.join(timeout=1.0)
