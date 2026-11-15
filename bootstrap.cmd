@echo off
setlocal
rem qwenstyle bootstrap launcher: unblock the local .ps1 (files downloaded from the internet carry
rem a zone marker) and run it with an explicit ExecutionPolicy - no manual policy changes needed.
powershell -NoProfile -Command "Get-ChildItem -Path '%~dp0' -Filter *.ps1 -ErrorAction SilentlyContinue | Unblock-File" >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" %*
exit /b %ERRORLEVEL%
