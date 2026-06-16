"""python -m hrfidelity serve"""
import argparse
import os
import uvicorn

from hrfidelity.tracing import setup_tracing


def main():
    parser = argparse.ArgumentParser(prog="hrfidelity")
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve", help="Start the certification dashboard")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    serve.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.cmd == "serve":
        setup_tracing()
        uvicorn.run(
            "hrfidelity.server.app:create_app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            factory=True,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
