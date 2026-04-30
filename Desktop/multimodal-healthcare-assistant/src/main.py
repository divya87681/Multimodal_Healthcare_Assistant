def main():
    print("Multimodal Healthcare Assistant Started")

    print("1. Pain Detection")
    print("2. Agitation Detection")
    print("3. Audio Emergency Detection")

    choice = input("Select module: ")

    if choice == "1":
        from icu_pain_watcher import run_pain_detector
        run_pain_detector()

    elif choice == "2":
        from icu_body_watcher import run_body_detector
        run_body_detector()

    elif choice == "3":
        from audio_emergency_detector import run_audio_detector
        run_audio_detector()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()