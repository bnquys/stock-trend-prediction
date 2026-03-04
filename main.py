import argparse
import sys


def run_train():
    print("\n" + "█"*60)
    print(" PHASE 1: HUẤN LUYỆN MÔ HÌNH xLSTM")
    print("█"*60)
    from train import main as train_main
    model, history, test_data, close_test = train_main()
    return model, test_data, close_test


def run_eval(model=None, test_data=None, close_test=None):
    print("\n" + "█"*60)
    print(" PHASE 2: ĐÁNH GIÁ MÔ HÌNH")
    print("█"*60)
    from evaluate import full_evaluation
    return full_evaluation(model, test_data, close_test)


def run_predict():
    print("\n" + "█"*60)
    print(" PHASE 3: DỰ BÁO PHIÊN TIẾP THEO")
    print("█"*60)
    from predict import predict_latest, run_backtest
    predict_latest()
    run_backtest()


def main():
    parser = argparse.ArgumentParser(
        description="xLSTM VNM Stock Trend Predictor"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["train", "eval", "predict", "all"],
        help="Chọn chế độ chạy"
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_train()

    elif args.mode == "eval":
        run_eval()

    elif args.mode == "predict":
        run_predict()

    elif args.mode == "all":
        model, test_data, close_test = run_train()
        run_eval(model, test_data, close_test)
        run_predict()

    print("\n Hoàn tất!")


if __name__ == "__main__":
    main()
