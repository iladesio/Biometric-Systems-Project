from src.recogniser.wbb_recogniser import WBBRecogniser


def main():
    wbb_recogniser = WBBRecogniser()

    wbb_recogniser.run_test(pathname="../data/Test/Walk_Ex_mavi_1")


if __name__ == "__main__":
    main()
