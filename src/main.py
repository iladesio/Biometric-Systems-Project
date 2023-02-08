from src.recogniser.wbb_recogniser import WBBRecogniser


def main():
    wbb_recogniser = WBBRecogniser()
    # wbb_recogniser.run_test(pathname="../data/Test/Walk_Ex_pippo_2")
    # wbb_recogniser.run_all_test()
    wbb_recogniser.perform_evaluation()

if __name__ == "__main__":
    main()
