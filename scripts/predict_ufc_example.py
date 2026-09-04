import _path_setup  # noqa: F401

from ufc_predictor.inference import predict_fight_winner


def main():
    prediction = predict_fight_winner(
        red_fighter_id="ad32471f01e7b1a5",
        blue_fighter_id="79899ecf62020f6d",
    )
    print(prediction)


if __name__ == "__main__":
    main()

