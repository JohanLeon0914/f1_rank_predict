# Comandos para ejecutar la API

## Ejecución local en PowerShell

```powershell
cd D:\f1_rank_predict

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

$env:PYTHONPATH = "src"
python scripts\run_api.py
```

La API queda disponible en `http://localhost:8000`.

## Verificar la API

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/metrics
```

Documentación interactiva:

```text
http://localhost:8000/docs
```

## Usar otro puerto

```powershell
$env:PYTHONPATH = "src"
python scripts\run_api.py --port 8010
```

## Ejecutar con Docker

```powershell
cd D:\f1_rank_predict

docker build -t f1-ranker-api:local .
docker run --rm -p 8080:8080 f1-ranker-api:local
```

Verificar el contenedor:

```powershell
Invoke-RestMethod http://localhost:8080/health
```

Documentación interactiva del contenedor:

```text
http://localhost:8080/docs
```
