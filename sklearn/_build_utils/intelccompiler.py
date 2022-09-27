"""Implements compiler object for Intel

Copyright (c) 2005-2022, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import platform
from setuptools._distutils.unixccompiler import UnixCCompiler


class IntelEM64TCCompiler(UnixCCompiler):
    """
    A modified Intel x86_64 compiler compatible with a 64bit GCC-built Python.
    """

    compiler_type = "intelem"
    cc_exe = "icc -m64"
    cc_args = "-fPIC"

    def __init__(self, verbose=0, dry_run=0, force=0):
        UnixCCompiler.__init__(self, verbose, dry_run, force)

        self.cc_exe = (
            "icc -std=c99 -m64 -fPIC -fp-model strict -O3 -fomit-frame-pointer -qopenmp"
        )
        compiler = self.cc_exe

        if platform.system() == "Darwin":
            shared_flag = "-Wl,-undefined,dynamic_lookup"
        else:
            shared_flag = "-shared"
        self.set_executables(
            compiler=compiler,
            compiler_so=compiler,
            compiler_cxx=compiler,
            archiver="xiar" + " cru",
            linker_exe=compiler + " -shared-intel",
            linker_so=compiler + " " + shared_flag + " -shared-intel",
        )
