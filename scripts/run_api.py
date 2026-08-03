import _path_setup  # noqa: F401
import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Levanta la API HTTP del F1 Ranker.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=int(os.getenv("PORT", "8000")), type=int)
    args = parser.parse_args()

    uvicorn.run("f1_ranker.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
