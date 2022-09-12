"""Patches new_compiler to supported compilers."""
from contextlib import suppress
from importlib import import_module

from setuptools.command.build_ext import new_compiler as orig_new_compiler

from .intelccompiler import IntelEM64TCCompiler


def patched_new_compiler(plat=None, compiler=None, verbose=0, dry_run=0, force=0):
    if compiler == "intelem":
        return IntelEM64TCCompiler(None, dry_run, force)
    return orig_new_compiler(
        plat=plat, compiler=compiler, verbose=verbose, dry_run=dry_run, force=force
    )


def patch_new_compiler():
    patch_modules = [
        "setuptools.command.build_ext",
        "distutils.ccompiler",
        "setuptools._distutils.ccompiler",
    ]

    for patch_module in patch_modules:
        with suppress(ImportError):
            mod = import_module(patch_module)
            mod.new_compiler = patch_new_compiler
