"""Tools for decoding Dynamic Routing Facemap data."""

__all__ = ["decode_all_sessions", "decode_session"]


def __getattr__(name: str):
    if name in __all__:
        from .decode_facemap import decode_all_sessions, decode_session

        return {"decode_all_sessions": decode_all_sessions, "decode_session": decode_session}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Run the all-session decoding command-line interface."""

    from .decode_facemap import main as decode_main

    decode_main()
