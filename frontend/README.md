# F1 ML Predicts Frontend

Frontend separado en Next.js, TypeScript y Tailwind para consumir la API local del modelo F1 Ranker.

## Comandos

Backend desde la raiz del repo:

```bash
.venv/bin/python scripts/run_api.py
```

Frontend:

```bash
cd frontend
npm run dev
```

Rutas:

- `/`: reservada para una landing futura.
- `/predicts`: simulacion libre de carreras.
- `/races`: calendario futuro y prediccion por carrera seleccionada.
- `/history`: historico de predicciones guardadas.

## Variables

Copia `.env.example` a `.env.local` y ajusta valores:

```bash
ML_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=
```

Si Supabase no esta configurado, las predicciones se guardan temporalmente en `localStorage`.
No uses `sb_secret_*` en el frontend. Esa llave es solo para backend, server functions o workers.

## Supabase

Proyecto esperado: `f1-ml-predicts`.

Ejecuta esta migracion en el SQL editor de Supabase:

```txt
supabase/migrations/20260828161000_create_prediction_tables.sql
```

Para modo invitado, la migracion habilita RLS y permite `select`/`insert` con la publishable key. No habilita update/delete. Cuando agregues login obligatorio, endurece las policies para filtrar por `auth.uid()`.
