@echo off
title Telegram Voice Assistant Engine
cd /d "C:\C_C++_C#\Vacancies\Voice-assistant"

echo ===================================================
echo  STARTING TELEGRAM BOT...
echo ===================================================

:: Если у тебя используется виртуальное окружение (venv), раскомментируй строку ниже (удали два двоеточия):
:: call venv\Scripts\activate

python main.py

pause