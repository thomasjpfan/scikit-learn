@rem https://github.com/numba/numba/blob/master/buildscripts/incremental/setup_conda_environment.cmd
@rem The cmd /C hack circumvents a regression where conda installs a conda.bat
@rem script in non-root environments.
set CONDA_INSTALL=cmd /C conda install -q -y
set PIP_INSTALL=pip install -q

@echo on

pip install typing-extensions

REM IF "%PYTHON_ARCH%"=="64" (
REM     @rem Deactivate any environment
REM     call deactivate
REM     @rem Clean up any left-over from a previous build
REM     conda remove --all -q -y -n %VIRTUALENV%
REM     conda create -n %VIRTUALENV% -q -y python=%PYTHON_VERSION% numpy scipy cython matplotlib wheel pillow joblib

REM     call activate %VIRTUALENV%

REM     pip install threadpoolctl

REM     IF "%PYTEST_VERSION%"=="*" (
REM         pip install pytest
REM     ) else (
REM         pip install pytest==%PYTEST_VERSION%
REM     )
REM ) else (
REM     pip install numpy scipy cython pytest wheel pillow joblib threadpoolctl
REM )

REM IF "%PYTEST_XDIST%" == "true" (
REM     pip install pytest-xdist
REM )

REM if "%COVERAGE%" == "true" (
REM     pip install coverage codecov pytest-cov
REM )
REM python --version
REM pip --version

REM @rem Install the build and runtime dependencies of the project.
REM python setup.py bdist_wheel bdist_wininst -b doc\logos\scikit-learn-logo.bmp

REM @rem Install the generated wheel package to test it
REM pip install --pre --no-index --find-links dist\ scikit-learn

REM if %errorlevel% neq 0 exit /b %errorlevel%
