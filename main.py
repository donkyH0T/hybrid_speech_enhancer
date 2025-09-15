from real_time_transcriber import RealTimeTranscriber

def main():
    transcriber = RealTimeTranscriber(model_path="./vosk-model-small-ru-0.22")
    print("Готово к записи и переводу речи...")
    try:
        transcriber.start()
    except KeyboardInterrupt:
        print("\nЗавершаем программу...")

if __name__ == "__main__":
    main()
