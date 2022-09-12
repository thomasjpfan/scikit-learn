"""Patches new_compiler to supported compilers."""
from setuptools.command.build_ext import new_compiler as orig_new_compiler
import setuptools.command.build_ext
from .intelccompiler import IntelEM64TCCompiler


def patched_new_compiler(plat=None, compiler=None, verbose=0, dry_run=0, force=0):
    if compiler == "intelem":
        return IntelEM64TCCompiler(None, dry_run, force)
    return orig_new_compiler(
        plat=plat, compiler=compiler, verbose=verbose, dry_run=dry_run, force=force
    )


def patch_new_compiler():
    setuptools.command.build_ext.new_compiler = patched_new_compiler
