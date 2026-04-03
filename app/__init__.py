# Patch httpx.AsyncClient to accept `app` argument for FastAPI testing
try:
    import httpx
    from httpx import AsyncClient, ASGITransport

    _original_init = AsyncClient.__init__

    def _patched_init(self, *args, app=None, **kwargs):
        if app is not None:
            # Provide an ASGITransport if none supplied
            if "transport" not in kwargs:
                kwargs["transport"] = ASGITransport(app=app)
        _original_init(self, *args, **kwargs)

    AsyncClient.__init__ = _patched_init
except Exception:
    # If httpx is not installed or any error occurs, fail silently – the app will still work.
    pass
