@echo off
chcp 65001 >nul

echo ╔══════════════════════════════════════════════╗
echo ║   Подбор аугментации — 10 конфигов           ║
echo ║   Модель: m4_deep_5block (зафиксирована)     ║
echo ╚══════════════════════════════════════════════╝
echo.

set PYTHON=python
set SCRIPT=train_stgcn_biisc.py

echo [01/10] no_aug — базовая точка без аугментации
%PYTHON% %SCRIPT% --config config_a01_no_aug.yaml
if %errorlevel% neq 0 echo [WARN] a01 завершился с ошибкой

echo.
echo [02/10] noise_only — только шум детекции
%PYTHON% %SCRIPT% --config config_a02_noise_only.yaml
if %errorlevel% neq 0 echo [WARN] a02 завершился с ошибкой

echo.
echo [03/10] temporal_only — только временные
%PYTHON% %SCRIPT% --config config_a03_temporal_only.yaml
if %errorlevel% neq 0 echo [WARN] a03 завершился с ошибкой

echo.
echo [04/10] cctv_only — только углы камеры
%PYTHON% %SCRIPT% --config config_a04_cctv_only.yaml
if %errorlevel% neq 0 echo [WARN] a04 завершился с ошибкой

echo.
echo [05/10] spatial_only — только пространственные
%PYTHON% %SCRIPT% --config config_a05_spatial_only.yaml
if %errorlevel% neq 0 echo [WARN] a05 завершился с ошибкой

echo.
echo [06/10] light_all — все группы, слабые параметры
%PYTHON% %SCRIPT% --config config_a06_light_all.yaml
if %errorlevel% neq 0 echo [WARN] a06 завершился с ошибкой

echo.
echo [07/10] medium_all — все группы, средние параметры
%PYTHON% %SCRIPT% --config config_a07_medium_all.yaml
if %errorlevel% neq 0 echo [WARN] a07 завершился с ошибкой

echo.
echo [08/10] heavy_all — все группы, агрессивные параметры
%PYTHON% %SCRIPT% --config config_a08_heavy_all.yaml
if %errorlevel% neq 0 echo [WARN] a08 завершился с ошибкой

echo.
echo [09/10] cctv_plus_noise — CCTV + шум (реалистично)
%PYTHON% %SCRIPT% --config config_a09_cctv_plus_noise.yaml
if %errorlevel% neq 0 echo [WARN] a09 завершился с ошибкой

echo.
echo [10/10] temporal_plus_cctv — temporal + CCTV без шума
%PYTHON% %SCRIPT% --config config_a10_temporal_plus_cctv.yaml
if %errorlevel% neq 0 echo [WARN] a10 завершился с ошибкой

echo.
echo ══════════════════════════════════════════════
echo  Готово! Открой TensorBoard:
echo    tensorboard --logdir runs
echo.
echo  Метрики для сравнения:
echo    F1_macro/val     — общее качество
echo    F1_val/COUG      — кашель
echo    F1_val/SNEE      — чихание
echo    Loss/train vs val — переобучение
echo ══════════════════════════════════════════════
pause
