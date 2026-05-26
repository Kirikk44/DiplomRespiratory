@echo off
chcp 65001 >nul

echo ╔══════════════════════════════════════════════╗
echo ║     Подбор архитектуры — 5 моделей           ║
echo ║     Аугментация: одинаковая во всех          ║
echo ╚══════════════════════════════════════════════╝
echo.

set PYTHON=python
set SCRIPT=train_stgcn_biisc.py

echo [1/5]  tiny    — 2 блока [32, 64]
%PYTHON% %SCRIPT% --config config_m1_tiny.yaml
if %errorlevel% neq 0 echo [WARN] m1 завершился с ошибкой

echo.
echo [2/5]  small   — 3 блока [64, 128, 256]
%PYTHON% %SCRIPT% --config config_m2_small.yaml
if %errorlevel% neq 0 echo [WARN] m2 завершился с ошибкой

echo.
echo [3/5]  medium  — 4 блока [64, 64, 128, 256]
%PYTHON% %SCRIPT% --config config_m3_medium.yaml
if %errorlevel% neq 0 echo [WARN] m3 завершился с ошибкой

echo.
echo [4/5]  deep    — 5 блоков [64, 64, 128, 128, 256] + stride
%PYTHON% %SCRIPT% --config config_m4_deep.yaml
if %errorlevel% neq 0 echo [WARN] m4 завершился с ошибкой

echo.
echo [5/5]  wide    — 3 блока [128, 256, 512]
%PYTHON% %SCRIPT% --config config_m5_wide.yaml
if %errorlevel% neq 0 echo [WARN] m5 завершился с ошибкой

echo.
echo ══════════════════════════════════════════════
echo  Готово! Сравни в TensorBoard:
echo    tensorboard --logdir runs
echo.
echo  Смотри на: F1_macro/val и F1_val/COUG + F1_val/SNEE
echo  Выбери лучшую модель и сообщи — сделаю 10 конфигов
echo  для разных аугментаций с зафиксированной архитектурой.
echo ══════════════════════════════════════════════
pause
