import _path_setup  # noqa: F401

from f1_ranker.data_loading import load_training_data
from f1_ranker.inference import predict_race_ranking


def main():
    """Predice una carrera existente tratandola como nueva para probar inferencia."""
    datasets = load_training_data()
    results = datasets["results"]
    races = datasets["races"][["raceId", "circuitId", "date"]]
    qualifying = datasets["qualifying"][
        ["raceId", "driverId", "constructorId", "position", "q1", "q2", "q3"]
    ]

    race_id = int(results["raceId"].max())
    race = races[races["raceId"] == race_id].iloc[0]

    participants = results[results["raceId"] == race_id][
        ["driverId", "constructorId", "grid"]
    ].merge(
        qualifying[qualifying["raceId"] == race_id],
        on=["driverId", "constructorId"],
        how="left",
    )
    participants = participants.rename(columns={"position": "qualifying_position"})

    prediction = predict_race_ranking(
        race_id=race_id,
        circuit_id=int(race["circuitId"]),
        participants=participants,
        race_date=str(race["date"]),
    )

    print(f"raceId: {race_id}")
    print(f"circuitId: {int(race['circuitId'])}")
    print(f"date: {race['date']}")
    print(prediction.to_string(index=False))


if __name__ == "__main__":
    main()
