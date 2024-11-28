try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("final_project")
except PackageNotFoundError:
    __version__ = "0.0.0"
