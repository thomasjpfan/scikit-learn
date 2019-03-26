@echo on

call activate %VIRTUALENV% || EXIT /B 1

copy %TMP_FOLDER%\.coverage %BUILD_REPOSITORY_LOCALPATH%

codecov --root %BUILD_REPOSITORY_LOCALPATH% -t %CODECOV_TOKEN%
