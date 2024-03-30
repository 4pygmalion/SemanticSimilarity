import argparse
import uvicorn


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        help="port number for service exposure.",
        type=int,
        default=8000,
    )
    parser.add_argument(
        "-w", "--workers", help="Number of workers", type=int, default=1
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = get_args()

    uvicorn.run(
        app="app:app",
        host="0.0.0.0",
        port=ARGS.port,
        reload=False,
        workers=ARGS.workers,
    )
