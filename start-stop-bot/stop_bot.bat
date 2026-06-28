@echo off
echo ===================================================
echo  STOPPING TELEGRAM BOT...
echo ===================================================

:: Эта команда жестко закроет окно консоли, где выполняется наш скрипт
taskkill /FI "WINDOWTITLE eq Telegram Voice Assistant Engine*" /F

:: На всякий случай гасим все процессы python, если бот ушел в фон (опционально)
:: taskkill /IM python.exe /F

echo Bot has been stopped successfully!
timeout /t 3